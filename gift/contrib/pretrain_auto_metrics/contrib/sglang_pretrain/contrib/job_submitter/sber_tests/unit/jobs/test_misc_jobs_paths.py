import json
from pathlib import Path

import pytest
from job_submitter.jobs.any_script import AnyScriptJob
from job_submitter.jobs.image_registry import PAM_BASE_IMAGES, load_pam_base_images
from job_submitter.jobs.infinite import InfiniteJob
from job_submitter.jobs.llmf.autometrics import AutometricsJobLLMF
from job_submitter.jobs.llmf.train import TrainJobLLMF
from job_submitter.jobs.pam.autometrics import AutometricsJobPAM
from job_submitter.jobs.pam.autometrics_external_api import (
    AutometricsJobPAMExternalAPI,
)
from job_submitter.jobs.verl.autometrics import AutometricsJobVERL
from job_submitter.jobs.verl.base import RayBaseJobVERL
from job_submitter.jobs.verl.cluster import RayClusterJobVERL
from job_submitter.jobs.verl.train import RayTrainJobVERL


@pytest.mark.unit
class TestNoopBeforeSubmitBranches:
    def test_any_script_before_submit(self) -> None:
        job = AnyScriptJob(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            script="echo 1",
        )
        assert job._before_submit() is None

    def test_infinite_before_submit(self) -> None:
        job = InfiniteJob(
            user_login="u", exp_name="e", submission_type="cloud", region="SR008"
        )
        assert job._before_submit() is None

    def test_ray_cluster_before_submit(self) -> None:
        job = RayClusterJobVERL(
            user_login="u", exp_name="e", submission_type="cloud", region="SR008"
        )
        assert job._before_submit() is None


@pytest.mark.unit
class TestLLMFTrainSubmitPath:
    def test_submit_calls_anonimize_config_then_submits_cfg(self, monkeypatch) -> None:
        called = {}
        fake_cfg = object()
        job = TrainJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            path_to_config="/tmp/train.yaml",
        )
        monkeypatch.setattr(
            job, "_anonimize_config", lambda p: called.setdefault("anonimize", p)
        )
        monkeypatch.setattr(job, "to_config", lambda: fake_cfg)
        monkeypatch.setattr(job, "assert_job", lambda job_obj, cfg: None)
        monkeypatch.setattr(job, "_before_submit", lambda: None)
        monkeypatch.setattr(job, "_submit_config", lambda cfg: "ok")
        assert job.submit() == "ok"
        assert called["anonimize"] == "/tmp/train.yaml"

    def test_train_external_libraries_and_before_submit(self) -> None:
        job = TrainJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            path_to_config="/tmp/train.yaml",
        )
        libs = job._external_code_libraries()
        assert len(libs) >= 4
        assert job._before_submit() is None

    def test_llmf_autometrics_external_libraries(self, tmp_path) -> None:
        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(tmp_path),
            sglang_tag="opensource",
        )
        libs = job._external_code_libraries()
        assert len(libs) >= 4
        assert not any("sglang_pretrain" in lib for lib in libs)

    def test_pam_autometrics_slurm_script_runs_node_runner(self, tmp_path) -> None:
        job = AutometricsJobPAM(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            checkpoint_dir=str(tmp_path),
        )
        script = job._script()
        assert script.startswith(f"cd {job.pam_root}/node_runner")
        assert f"--checkpoint {tmp_path}" in script
        assert f"--mode {job.mode}" in script

    def test_pam_autometrics_cloud_script_passes_wdir(self, tmp_path) -> None:
        job = AutometricsJobPAM(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(tmp_path),
        )
        assert job._script().startswith(f"--wdir {job.pam_root}/node_runner")

    def test_pam_autometrics_external_libraries(self, tmp_path) -> None:
        job = AutometricsJobPAM(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(tmp_path),
            sglang_tag="opensource",
        )
        libs = job._external_code_libraries()
        assert len(libs) == 4
        assert not any("sglang_pretrain" in lib for lib in libs)

    def test_pam_autometrics_custom_sglang_tag_adds_sglang(self, tmp_path) -> None:
        job = AutometricsJobPAM(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(tmp_path),
            sglang_tag="custom",
        )
        libs = job._external_code_libraries()
        suffix = Path("contrib/sglang_pretrain/python").as_posix()
        assert any(Path(lib).as_posix().endswith(suffix) for lib in libs)

    def test_pam_external_api_script_uses_config(self, tmp_path) -> None:
        job = AutometricsJobPAMExternalAPI(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR004",
            path_to_config=str(tmp_path / "cfg.yaml"),
        )
        script = job._script()
        assert script.startswith(f"--wdir {job.pam_root}/node_runner")
        assert f"--config {tmp_path / 'cfg.yaml'}" in script
        assert "--checkpoint" not in script
        assert job._before_submit() is None

    def test_pam_external_api_script_adds_output_dir(self, tmp_path) -> None:
        job = AutometricsJobPAMExternalAPI(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR004",
            path_to_config=str(tmp_path / "cfg.yaml"),
            output_dir=str(tmp_path / "out"),
        )
        assert f"--checkpoint {tmp_path / 'out'}" in job._script()

    def test_pam_external_api_rejects_slurm(self, tmp_path) -> None:
        job = AutometricsJobPAMExternalAPI(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            path_to_config=str(tmp_path / "cfg.yaml"),
        )
        with pytest.raises(ValueError, match="Slurm is not supported"):
            job._script()

    def test_pam_external_api_resolves_base_image(self) -> None:
        job = AutometricsJobPAMExternalAPI(
            user_login="u", exp_name="e", submission_type="cloud", region="SR004"
        )
        assert job._resolve_base_image() == job.BASE_IMAGES["custom"]["SR004"]

        explicit = AutometricsJobPAMExternalAPI(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR004",
            base_image="/img.sqsh",
        )
        assert explicit._resolve_base_image() == "/img.sqsh"

    def test_pam_external_api_rejects_unknown_sglang_tag(self) -> None:
        job = AutometricsJobPAMExternalAPI(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR004",
            sglang_tag="nope",
        )
        with pytest.raises(AssertionError):
            job._resolve_base_image()

    def test_pam_external_api_external_libraries(self) -> None:
        job = AutometricsJobPAMExternalAPI(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR004",
            sglang_tag="opensource",
        )
        assert len(job._external_code_libraries()) == 4

        custom = AutometricsJobPAMExternalAPI(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR004",
            sglang_tag="custom",
        )
        suffix = Path("contrib/sglang_pretrain/python").as_posix()
        assert any(
            Path(lib).as_posix().endswith(suffix)
            for lib in custom._external_code_libraries()
        )

    def test_llmf_autometrics_external_libraries_custom_sglang_tag_adds_sglang(
        self, tmp_path
    ) -> None:
        job = AutometricsJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(tmp_path),
            sglang_tag="custom",
        )
        libs = job._external_code_libraries()
        suffix = Path(
            "contrib/pretrain_auto_metrics/contrib/sglang_pretrain/python"
        ).as_posix()
        assert any(Path(p).as_posix().endswith(suffix) for p in libs)


PAM_CHECKOUT_IMAGES = {
    "custom": {"SR008": "cr.ai.cloud.ru/test/pam_custom:v1.0.0"},
    "opensource": {"SR008": "cr.ai.cloud.ru/test/pam_opensource:v1.0.0"},
}


@pytest.fixture
def pam_checkout(tmp_path, monkeypatch) -> Path:
    (tmp_path / "images.json").write_text(json.dumps(PAM_CHECKOUT_IMAGES))
    for job_cls in (AutometricsJobPAM, AutometricsJobPAMExternalAPI):
        monkeypatch.setattr(job_cls, "pam_root", property(lambda self: tmp_path))
    return tmp_path


@pytest.mark.unit
class TestPamCheckoutImages:
    def test_load_pam_base_images_reads_checkout_registry(self, pam_checkout) -> None:
        assert load_pam_base_images(pam_checkout) == PAM_CHECKOUT_IMAGES

    def test_load_pam_base_images_falls_back_without_registry(self, tmp_path) -> None:
        assert load_pam_base_images(tmp_path) is PAM_BASE_IMAGES

    @pytest.mark.parametrize(
        "job",
        [
            AutometricsJobPAM(
                user_login="u",
                exp_name="e",
                submission_type="cloud",
                region="SR008",
                checkpoint_dir="/ckpt",
                sglang_tag="opensource",
            ),
            AutometricsJobPAMExternalAPI(
                user_login="u",
                exp_name="e",
                submission_type="cloud",
                region="SR008",
                sglang_tag="opensource",
            ),
        ],
    )
    def test_pam_jobs_resolve_image_from_checkout(self, pam_checkout, job) -> None:
        assert job._resolve_base_image() == "cr.ai.cloud.ru/test/pam_opensource:v1.0.0"


@pytest.mark.unit
class TestVerlSpecificBranches:
    def test_autometrics_verl_root_uses_env(self, monkeypatch) -> None:
        monkeypatch.setenv("VERL_ROOT_PATH", "/tmp/verl-root")
        job = AutometricsJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir="/tmp",
        )
        assert job.verl_root == Path("/tmp/verl-root")
        script = job._script()
        assert "node_runner.py" in script
        assert script.startswith("--wdir ")

    def test_verl_train_build_env_adds_runtime_keys(
        self, monkeypatch, tmp_path
    ) -> None:
        cfg = tmp_path / "runtime.yaml"
        cfg.write_text("excludes: []\n", encoding="utf-8")
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            script_path="/tmp/run.sh",
            runtime_env_config_path=str(cfg),
        )
        monkeypatch.setattr(
            job, "_replace_runtime_env_config", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(job, "slurm_jwt_token", "tkn")
        monkeypatch.setattr(job, "slurmrestd_urls", ["http://a:6820"])
        env = job._build_env()
        assert env["RUNTIME_ENV"] == str(cfg)
        assert env["TRAINING_SCRIPT"] == "/tmp/run.sh"

    def test_verl_autometrics_default_root_and_slurm_script(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.delenv("VERL_ROOT_PATH", raising=False)
        job = AutometricsJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            checkpoint_dir=str(tmp_path),
            sglang_tag="opensource",
        )
        assert "python3 node_runner.py" in job._script()
        libs = job._external_code_libraries()
        assert len(libs) >= 4
        assert not any("sglang_pretrain" in lib for lib in libs)

    def test_verl_autometrics_external_libraries_custom_sglang_tag_adds_sglang(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("VERL_ROOT_PATH", "/tmp/verl-root")
        job = AutometricsJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            checkpoint_dir=str(tmp_path),
            sglang_tag="custom",
        )
        libs = job._external_code_libraries()
        suffix = Path(
            "sber_src/contribs/pretrain_auto_metrics/contrib/sglang_pretrain/python"
        ).as_posix()
        assert any(Path(p).as_posix().endswith(suffix) for p in libs)

    def test_ray_base_resolve_base_image_and_build_env(self, monkeypatch) -> None:
        job = RayClusterJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            engine="vllm",
            use_sharp=True,
        )
        monkeypatch.setattr(
            "job_submitter.jobs.verl.base.BaseJob._build_env", lambda self: {}
        )
        assert "verl_0.7.1" in job._resolve_base_image()
        env = RayBaseJobVERL._build_env(job)
        assert env["NCCL_COLLNET_ENABLE"] == "1"

    def test_verl_train_cloud_script_and_no_pam_config_branch(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("VERL_ROOT_PATH", "/tmp/verl-root")
        cfg = tmp_path / "runtime.yaml"
        cfg.write_text("x: 1\n", encoding="utf-8")
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            script_path="/tmp/run.sh",
            runtime_env_config_path=str(cfg),
            pam_config_path=None,
        )
        script = job._script()
        assert script.startswith("bash ")
        rest = script[len("bash ") :].strip()
        assert rest.replace("\\", "/").endswith(
            "/tmp/verl-root/sber_src/submit/setup_ray_cluster/cloud.sh"
        )
