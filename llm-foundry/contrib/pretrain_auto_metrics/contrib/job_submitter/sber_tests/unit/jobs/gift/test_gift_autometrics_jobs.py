from pathlib import Path

import pytest

from job_submitter.jobs.gift import autometrics as gift_autometrics_module
from job_submitter.jobs.gift import (
    autometrics_external_api as gift_external_api_module,
)
from job_submitter.jobs.gift.autometrics import AutometricsJobGIFT
from job_submitter.jobs.gift.autometrics_external_api import (
    AutometricsJobGIFTExternalAPI,
)

pytestmark = pytest.mark.unit


def _gift_root(tmp_path: Path) -> Path:
    root = tmp_path / "torchtitan checkout"
    node_runner = (
        root / "contrib" / "pretrain_auto_metrics" / "node_runner" / "node_runner.py"
    )
    node_runner.parent.mkdir(parents=True)
    node_runner.write_text("pass\n", encoding="utf-8")
    return root


def _hf_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "hf checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    return checkpoint


def _pam_config(tmp_path: Path) -> Path:
    config = tmp_path / "pam.yaml"
    config.write_text(
        """\
run_name: gift-test
exp_name: gift-test
user: autotest
tensorboard_path: /tmp/tensorboard
max_seq_len: 4096
autometrics:
  globals:
    delete_converted_hf: false
  router:
    host: 0.0.0.0
  convertation:
    mappers:
      - map_gigafsdp_to_hf_safetensor
    input_tp_size: 1
  preparation:
    skip: true
  inference:
    tp: 8
  model_info:
    model_class: sft
    size: custom
    username: autotest
  evaluators: {}
""",
        encoding="utf-8",
    )
    return config


def _external_api_config(tmp_path: Path) -> Path:
    config = tmp_path / "external.yaml"
    config.write_text(
        """\
run_name: gift-external-api-test
exp_name: gift-external-api-test
user: autotest
autometrics:
  inference:
    engine_type: external_api
    base_url: https://example.invalid/v1
    model: gift
    api_key: test-key
  evaluators:
    lm_eval: {}
""",
        encoding="utf-8",
    )
    return config


class TestAutometricsJobGIFT:
    def test_gift_root_uses_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = tmp_path / "gift from env"
        monkeypatch.setenv("GIFT_ROOT_PATH", str(root))
        job = AutometricsJobGIFT(
            user_login="u",
            exp_name="e",
            checkpoint_dir=str(tmp_path / "checkpoint"),
        )

        assert job.gift_root == root.resolve()

    def test_gift_root_defaults_to_vendored_checkout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("GIFT_ROOT_PATH", raising=False)
        job = AutometricsJobGIFT(
            user_login="u",
            exp_name="e",
            checkpoint_dir=str(tmp_path / "checkpoint"),
        )

        assert (
            job.gift_root == Path(gift_autometrics_module.__file__).resolve().parents[5]
        )

    @pytest.mark.parametrize(
        ("submission_type", "prefix"),
        [("cloud", "--wdir "), ("slurm", "cd ")],
    )
    def test_script_runs_vendored_pam_node_runner(
        self,
        submission_type: str,
        prefix: str,
        tmp_path: Path,
    ) -> None:
        root = _gift_root(tmp_path)
        checkpoint = _hf_checkpoint(tmp_path)
        job = AutometricsJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type=submission_type,
            region="SR008" if submission_type == "cloud" else "o1",
            checkpoint_dir=str(checkpoint),
            gift_root_path=root,
        )

        script = job._script()

        assert script.startswith(prefix)
        assert "pretrain_auto_metrics" in script
        assert "python3 node_runner.py" in script
        assert f"--checkpoint '{checkpoint}'" in script
        assert "--mode autometrics" in script

    def test_external_libraries_follow_pam_layout(self, tmp_path: Path) -> None:
        root = _gift_root(tmp_path)
        job = AutometricsJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(_hf_checkpoint(tmp_path)),
            gift_root_path=root,
            sglang_tag="custom",
        )

        libraries = [Path(path).as_posix() for path in job._external_code_libraries()]

        assert libraries[0] == root.resolve().as_posix()
        assert libraries[-1].endswith("contrib/sglang_pretrain/python")

    def test_external_libraries_opensource_tag_skips_sglang(
        self, tmp_path: Path
    ) -> None:
        job = AutometricsJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(_hf_checkpoint(tmp_path)),
            gift_root_path=_gift_root(tmp_path),
            sglang_tag="opensource",
        )

        libraries = job._external_code_libraries()

        assert not any("sglang_pretrain" in path for path in libraries)

    @pytest.mark.parametrize(
        ("missing_path", "message"),
        [("config.json", "config.json"), ("model.safetensors", "safetensors")],
    )
    def test_before_submit_rejects_incomplete_hf_checkpoint(
        self,
        missing_path: str,
        message: str,
        tmp_path: Path,
    ) -> None:
        root = _gift_root(tmp_path)
        checkpoint = _hf_checkpoint(tmp_path)
        (checkpoint / missing_path).unlink()
        job = AutometricsJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(checkpoint),
            gift_root_path=root,
        )

        with pytest.raises(ValueError, match=message):
            job._before_submit()

    def test_before_submit_accepts_raw_dcp_checkpoint(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "dcp checkpoint"
        checkpoint.mkdir()
        (checkpoint / ".metadata").write_bytes(b"dcp")
        job = AutometricsJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(checkpoint),
            path_to_config=str(_pam_config(tmp_path)),
            gift_root_path=_gift_root(tmp_path),
        )

        assert job._before_submit() is None

    def test_before_submit_rejects_missing_pam_checkout(self, tmp_path: Path) -> None:
        checkpoint = _hf_checkpoint(tmp_path)
        job = AutometricsJobGIFT(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(checkpoint),
            gift_root_path=tmp_path / "missing",
        )

        with pytest.raises(ValueError, match="node_runner"):
            job._before_submit()

    @pytest.mark.parametrize(
        ("submission_type", "region"),
        [("cloud", "SR008"), ("slurm", "o1")],
    )
    def test_dry_run_validates_complete_pam_contract(
        self,
        submission_type: str,
        region: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        checkpoint = _hf_checkpoint(tmp_path)
        job = AutometricsJobGIFT(
            user_login="autotest",
            exp_name="gift-pam",
            submission_type=submission_type,
            region=region,
            queue_name="pretrain" if submission_type == "slurm" else None,
            workspace_name="pretrain" if submission_type == "cloud" else None,
            checkpoint_dir=str(checkpoint),
            path_to_config=str(_pam_config(tmp_path)),
            gift_root_path=_gift_root(tmp_path),
            dry_run=True,
        )
        if submission_type == "slurm":
            monkeypatch.setattr(
                job, "_resolve_slurm_jwt_token", lambda: "sanitized-token"
            )
            monkeypatch.setattr(
                job,
                "_resolve_slurmrestd_urls",
                lambda: ["http://slurm.invalid:6820"],
            )

        result = job.submit()

        assert result is job
        assert job._last_submitted_job is None
        assert job.exp_name.endswith(" | autometrics mode")
        assert (checkpoint / "artifacts" / "pam_config.yaml").is_file()


class TestAutometricsJobGIFTExternalAPI:
    def test_gift_root_uses_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = tmp_path / "gift from env"
        monkeypatch.setenv("GIFT_ROOT_PATH", str(root))
        job = AutometricsJobGIFTExternalAPI(
            user_login="u",
            exp_name="e",
            path_to_config=str(tmp_path / "external.yaml"),
        )

        assert job.gift_root == root.resolve()

    def test_gift_root_defaults_to_vendored_checkout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("GIFT_ROOT_PATH", raising=False)
        job = AutometricsJobGIFTExternalAPI(
            user_login="u",
            exp_name="e",
            path_to_config=str(tmp_path / "external.yaml"),
        )

        assert (
            job.gift_root
            == Path(gift_external_api_module.__file__).resolve().parents[5]
        )

    def test_uses_vendored_pam_checkout(self, tmp_path: Path) -> None:
        root = _gift_root(tmp_path)
        config_path = tmp_path / "external.yaml"
        job = AutometricsJobGIFTExternalAPI(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            path_to_config=str(config_path),
            gift_root_path=root,
        )

        assert job.pam_root == root.resolve() / "contrib" / "pretrain_auto_metrics"
        assert job._script().startswith(f"--wdir {job.pam_root}/node_runner")
        assert f"--config {config_path}" in job._script()
        assert job._before_submit() is None

    def test_before_submit_rejects_missing_pam_checkout(self, tmp_path: Path) -> None:
        job = AutometricsJobGIFTExternalAPI(
            user_login="u",
            exp_name="e",
            path_to_config=str(tmp_path / "external.yaml"),
            gift_root_path=tmp_path / "missing",
        )

        with pytest.raises(ValueError, match="node_runner"):
            job._before_submit()

    def test_dry_run_validates_external_api_contract(self, tmp_path: Path) -> None:
        job = AutometricsJobGIFTExternalAPI(
            user_login="autotest",
            exp_name="gift-external-api",
            submission_type="cloud",
            region="SR008",
            workspace_name="pretrain",
            path_to_config=str(_external_api_config(tmp_path)),
            gift_root_path=_gift_root(tmp_path),
            dry_run=True,
        )

        result = job.submit()

        assert result is job
        assert job._last_submitted_job is None
