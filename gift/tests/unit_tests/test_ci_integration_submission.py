import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import submit_integration_job
from scripts.ci import verify_checkout


REPO_ROOT = "/workspace-SR008.nfs2/user/gitlab-builds/123/torchtitan"
EXPECTED_COMMIT = "a" * 40


def _stub_job_submitter_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    class BaseJob:
        def _before_submit(self) -> None:
            pass

        def _build_env(self) -> dict[str, str]:
            return {"key": "value"}

    job_submitter_module = types.ModuleType("job_submitter")
    jobs_module = types.ModuleType("job_submitter.jobs")
    base_module = types.ModuleType("job_submitter.jobs.base")
    test_job_module = types.ModuleType("test_job")
    base_module.BaseJob = BaseJob
    test_job_module.TestJobTorchTitan = type("TestJobTorchTitan", (), {})

    monkeypatch.setitem(sys.modules, "job_submitter", job_submitter_module)
    monkeypatch.setitem(sys.modules, "job_submitter.jobs", jobs_module)
    monkeypatch.setitem(sys.modules, "job_submitter.jobs.base", base_module)
    monkeypatch.setitem(sys.modules, "test_job", test_job_module)


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suite: str,
    image: str | None = "test-image",
) -> list[dict[str, Any]]:
    _stub_job_submitter_modules(monkeypatch)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    argv = [
        "submit_integration_job.py",
        "--suite",
        suite,
        "--ngpu",
        "8",
        "--output-dir",
        "artifacts/integration",
        "--artifact-dir",
        str(tmp_path),
    ]
    if image is not None:
        argv += ["--image", image]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("CI_COMMIT_SHA", EXPECTED_COMMIT)
    monkeypatch.setenv("CLUSTER_KEY", "SR008")
    monkeypatch.delenv("IMAGE_PATH", raising=False)
    monkeypatch.delenv("TORCHTITAN_INTEGRATION_IMAGE", raising=False)
    monkeypatch.delenv("TORCHTITAN_CLOUD_REGION", raising=False)

    submissions = []

    def submit_and_monitor(**kwargs: Any) -> str:
        submissions.append(kwargs)
        return "Completed"

    monkeypatch.setattr(
        submit_integration_job,
        "_submit_and_monitor",
        submit_and_monitor,
    )

    assert submit_integration_job.main() == 0
    return submissions


def test_remote_script_uses_current_checkout() -> None:
    script = submit_integration_job._build_remote_script(
        repo_root=REPO_ROOT,
        expected_commit=EXPECTED_COMMIT,
        suite="smoke",
        output_dir="artifacts/integration",
        ngpu=1,
    )

    assert script.startswith(f"--wdir {REPO_ROOT} python3 ")
    assert f"{REPO_ROOT}/scripts/ci/verify_checkout.py" in script
    assert f"--repo-root {REPO_ROOT}" in script
    assert f"--expected-commit {EXPECTED_COMMIT}" in script
    assert f"{REPO_ROOT}/tests/integration_tests/run_tests.py" in script
    assert "/home/user/torchtitan" not in script


def test_torchcomms_scripts_use_current_checkout() -> None:
    scripts = submit_integration_job._build_torchcomms_scripts(
        repo_root=REPO_ROOT,
        expected_commit=EXPECTED_COMMIT,
        output_dir="artifacts/integration",
    )

    assert len(scripts) == len(submit_integration_job.TORCHCOMMS_TESTS)
    config_files = {
        case[0]: case[2] for case in submit_integration_job.TORCHCOMMS_TESTS
    }
    for test_name, _, script in scripts:
        assert script.startswith(f"{REPO_ROOT}/scripts/ci/verify_checkout.py ")
        assert f"--repo-root {REPO_ROOT}" in script
        assert f"--expected-commit {EXPECTED_COMMIT}" in script
        # The YAML entrypoint, then the experiment file, then the overrides.
        assert f"{REPO_ROOT}/torchtitan/train.py {config_files[test_name]} " in script
        assert "--module" not in script and "--config " not in script
        assert f"--dump_folder {REPO_ROOT}/artifacts/integration/{test_name}" in script
        assert "/home/user/torchtitan" not in script


def test_torchcomms_experiment_files_exist_and_pass_pre_submit() -> None:
    for _, _, config_file, _ in submit_integration_job.TORCHCOMMS_TESTS:
        assert (submit_integration_job.REPO_ROOT / config_file).is_file(), config_file
    submit_integration_job._check_experiment_configs(
        [case[2] for case in submit_integration_job.TORCHCOMMS_TESTS]
    )


def test_extra_env_leaves_pythonpath_to_job_submitter() -> None:
    env = submit_integration_job._extra_env(
        repo_root=REPO_ROOT,
        expected_commit=EXPECTED_COMMIT,
        suite="features",
        output_dir="artifacts/integration",
        ngpu=8,
    )

    assert "PYTHONPATH" not in env
    assert env["TORCHTITAN_REPO_ROOT"] == REPO_ROOT
    assert env["TORCHTITAN_EXPECTED_COMMIT"] == EXPECTED_COMMIT


def test_artifact_dir_must_not_populate_test_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "test-output"

    with pytest.raises(ValueError, match="outside output-dir"):
        submit_integration_job._validate_artifact_dir(
            output_dir=str(output_dir),
            artifact_dir=output_dir,
        )
    with pytest.raises(ValueError, match="outside output-dir"):
        submit_integration_job._validate_artifact_dir(
            output_dir=str(output_dir),
            artifact_dir=output_dir / "cloud-job",
        )

    submit_integration_job._validate_artifact_dir(
        output_dir=str(output_dir),
        artifact_dir=tmp_path / "cloud-job",
    )


def test_all_ci_jobs_use_separate_cloud_artifact_dir() -> None:
    config = (submit_integration_job.REPO_ROOT / ".gitlab-ci.yml").read_text(
        encoding="utf-8"
    )

    assert '--artifact-dir "$OUTPUT_DIR"' not in config
    assert config.count('--artifact-dir "$TORCHTITAN_CLOUD_ARTIFACT_DIR"') == 5
    assert (
        'TEST_ARTIFACTS_PATH: "$TORCHTITAN_INTEGRATION_OUTPUT_DIR/cloud-job/"' in config
    )
    assert 'TEST_ARTIFACTS_PATH: "$TORCHTITAN_INTEGRATION_OUTPUT_DIR/"' not in config


@pytest.mark.parametrize("remote_checkout", [False, True])
def test_job_accepts_checkout_and_redacts_environment(
    monkeypatch,
    tmp_path: Path,
    remote_checkout: bool,
) -> None:
    _stub_job_submitter_modules(monkeypatch)
    module_name = "test_job_with_stubbed_base"
    module_path = submit_integration_job.REPO_ROOT / "scripts/ci/test_job.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    repo_root = tmp_path / "remote-checkout" if remote_checkout else tmp_path
    assert repo_root.exists() is not remote_checkout
    job = module.TestJobTorchTitan(script="true", repo_root=str(repo_root))
    job._before_submit()
    assert job._external_code_libraries()[0] == str(repo_root)
    env = job._build_env()

    assert repr(env) == "<redacted>"
    assert json.loads(json.dumps(env)) == {"key": "value"}


def test_main_defaults_image_to_pinned_registry(monkeypatch, tmp_path: Path) -> None:
    submissions = _run_main(monkeypatch, tmp_path, "smoke", image=None)

    pinned = json.loads(
        submit_integration_job.GIFT_IMAGES_PATH.read_text(encoding="utf-8")
    )
    assert len(submissions) == 1
    assert submissions[0]["image"] == pinned["SR008"]


def test_explicit_image_overrides_pinned_registry() -> None:
    assert submit_integration_job._resolve_image("custom-image", "SR008") == (
        "custom-image"
    )


def test_resolve_image_rejects_unpinned_region() -> None:
    with pytest.raises(SystemExit, match="SR999"):
        submit_integration_job._resolve_image(None, "SR999")


def test_ci_config_pins_no_integration_image() -> None:
    config = (submit_integration_job.REPO_ROOT / ".gitlab-ci.yml").read_text(
        encoding="utf-8"
    )

    assert "TORCHTITAN_INTEGRATION_IMAGE" not in config
    assert "torchtitan:2026" not in config


def test_resolve_expected_commit_prefers_ci_sha(monkeypatch) -> None:
    monkeypatch.setenv("CI_COMMIT_SHA", EXPECTED_COMMIT)

    assert (
        submit_integration_job._resolve_expected_commit(Path("unused"))
        == EXPECTED_COMMIT
    )


def test_verify_checkout_rejects_wrong_commit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        verify_checkout,
        "_git_head",
        lambda repo_root: "b" * 40,
    )

    with pytest.raises(RuntimeError, match="checkout mismatch"):
        verify_checkout._verify_checkout(tmp_path, EXPECTED_COMMIT)


@pytest.mark.parametrize("suite", ["smoke", "features", "models", "h100"])
@pytest.mark.parametrize("remote_root", [None, REPO_ROOT])
def test_main_submits_standard_suite(
    monkeypatch,
    tmp_path: Path,
    suite: str,
    remote_root: str | None,
) -> None:
    if remote_root is None:
        monkeypatch.delenv("SR008_REMOTE_DIR", raising=False)
    else:
        monkeypatch.setenv("SR008_REMOTE_DIR", remote_root)
    submissions = _run_main(monkeypatch, tmp_path, suite)

    assert len(submissions) == 1
    submission = submissions[0]
    expected_root = remote_root or str(submit_integration_job.REPO_ROOT)
    assert submission["repo_root"] == expected_root
    assert submission["script"].startswith(f"--wdir {expected_root} ")
    assert submission["extra_env"]["TORCHTITAN_REPO_ROOT"] == expected_root
    assert submission["suite"] == suite
    assert submission["extra_env"]["TORCHTITAN_EXPECTED_COMMIT"] == EXPECTED_COMMIT
    assert "/home/user/torchtitan" not in submission["script"]


@pytest.mark.parametrize("remote_root", [None, REPO_ROOT])
def test_main_submits_all_torchcomms_tests(
    monkeypatch, tmp_path: Path, remote_root: str | None
) -> None:
    if remote_root is None:
        monkeypatch.delenv("SR008_REMOTE_DIR", raising=False)
    else:
        monkeypatch.setenv("SR008_REMOTE_DIR", remote_root)
    submissions = _run_main(monkeypatch, tmp_path, "features_torchcomms")

    assert len(submissions) == len(submit_integration_job.TORCHCOMMS_TESTS)
    for submission in submissions:
        test_name = submission["extra_env"]["TORCHTITAN_INTEGRATION_TEST_NAME"]
        expected_trace_dir = submit_integration_job._remote_output_dir(
            remote_root or str(submit_integration_job.REPO_ROOT),
            "artifacts/integration",
            test_name,
        )
        assert submission["extra_env"]["TORCH_TRACE"] == (
            expected_trace_dir + "/compile_trace"
        )
        assert submission["script_type"] == "pytorch2"
        assert submission["processes_per_worker"] == 8


def test_preflight_checks_the_actual_overrides_and_planned_process_count(monkeypatch):
    import torchtitan.config

    seen = []
    original = torchtitan.config.run_pre_submit

    def checked(path, overrides=None, *, world_size=None):
        seen.append((overrides, world_size))
        return original(path, overrides, world_size=world_size)

    monkeypatch.setattr(torchtitan.config, "run_pre_submit", checked)
    with pytest.raises(SystemExit, match="invalid experiment config"):
        submit_integration_job._check_experiment_configs(
            ["configs/experiments/llama3_debug.yaml"],
            raw_overrides=[("parallelism.tensor_parallel_degree", "2")],
            world_size=1,
        )
    assert seen == [([("parallelism.tensor_parallel_degree", "2")], 1)]


def test_torchcomms_preflight_parses_each_tests_overrides(monkeypatch, tmp_path):
    import torchtitan.config

    seen = []
    original = torchtitan.config.run_pre_submit

    def checked(path, overrides=None, *, world_size=None):
        seen.append((Path(path).name, overrides, world_size))
        return original(path, overrides, world_size=world_size)

    monkeypatch.setattr(torchtitan.config, "run_pre_submit", checked)
    _run_main(monkeypatch, tmp_path, "features_torchcomms")
    assert [(name, world) for name, _, world in seen] == [
        (Path(case[2]).name, 8) for case in submit_integration_job.TORCHCOMMS_TESTS
    ]
    assert ("parallelism.context_parallel_degree", "2") in seen[0][1]
    assert ("compile.enable", "true") in seen[1][1]


@pytest.mark.parametrize("backend", ["deepep", "hybridep"])
def test_h100_token_drop_numerics_runs_assertions_module(
    monkeypatch, tmp_path, backend
):
    import subprocess

    from tests.integration_tests import run_tests
    from tests.integration_tests.h100 import build_h100_tests_list

    row = next(
        row
        for row in build_h100_tests_list()
        if row.test_name == f"token_drop_numerics_{backend}"
    )
    commands = []

    def run(cmd, timeout):
        commands.append(cmd)
        assert timeout == 600
        return subprocess.CompletedProcess(cmd, 0, stdout="PASS")

    monkeypatch.setattr(run_tests, "_run_cmd", run)
    run_tests.run_single_test(row, str(tmp_path), gpu_ids=[2, 3])
    assert len(commands) == 1
    command = commands[0]
    assert "--module tests.integration_tests.token_drop_numerics" in command
    assert "--nproc-per-node=2" in command
    assert "CUDA_VISIBLE_DEVICES=2,3" in command
    assert f"--backend {backend}" in command
    assert "--dump_folder" in command
    assert "run_train.sh" not in command
