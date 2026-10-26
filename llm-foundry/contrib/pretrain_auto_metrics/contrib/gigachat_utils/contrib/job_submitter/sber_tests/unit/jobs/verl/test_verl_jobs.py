from pathlib import Path

import pytest
from job_submitter.jobs.image_registry import VERL_BASE_IMAGES
from job_submitter.jobs.verl.cluster import RayClusterJobVERL
from job_submitter.jobs.verl.train import RayTrainJobVERL
from omegaconf import OmegaConf
from sber_tests.dry_run_helpers import assert_dry_run_submit_contract


@pytest.mark.unit
class TestRayClusterJobVERL:
    def test_cloud_submission_selects_cloud_script(self) -> None:
        job = RayClusterJobVERL(
            user_login="u", exp_name="e", submission_type="cloud", region="SR008"
        )
        assert "cloud.sh" in job._script()

    def test_slurm_submission_is_not_supported_yet(self) -> None:
        job = RayClusterJobVERL(
            user_login="u", exp_name="e", submission_type="slurm", region="o1"
        )
        with pytest.raises(AssertionError, match="not supported for slurm"):
            job._script()

    def test_build_env_adds_sharp_flag_when_enabled(self) -> None:
        job = RayClusterJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            use_sharp=True,
        )
        env = job._build_env()
        assert env["NCCL_COLLNET_ENABLE"] == "1"

    def test_build_env_does_not_add_sharp_flag_when_disabled(self) -> None:
        job = RayClusterJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            use_sharp=False,
        )
        env = job._build_env()
        assert "NCCL_COLLNET_ENABLE" not in env

    def test_build_env_outlasts_the_ray_stop_budget(self) -> None:
        # `ray stop --grace-period 15` plus a 5s forced stop, starting ~4s in.
        # The cleanup's per-process TERM fan-out must not land before that ends.
        job = RayClusterJobVERL(
            user_login="u", exp_name="e", submission_type="cloud", region="SR008"
        )
        assert int(job._build_env()["PROCESS_SUPERVISOR_GRACE_S"]) > 24

    def test_build_env_stages_fit_the_kill_wait_window(self) -> None:
        # Slurm SIGKILLs the step, wrapper included, at KillWait (commonly 60s).
        # The trailing 5 is the template's default KILL stage.
        job = RayClusterJobVERL(
            user_login="u", exp_name="e", submission_type="cloud", region="SR008"
        )
        env = job._build_env()
        stages = (
            int(env["PROCESS_SUPERVISOR_GRACE_S"])
            + int(env["PROCESS_SHUTDOWN_GRACE_S"])
            + 5
        )
        assert stages <= 60

    @pytest.mark.parametrize(
        "name", ["PROCESS_SUPERVISOR_GRACE_S", "PROCESS_SHUTDOWN_GRACE_S"]
    )
    def test_build_env_keeps_an_explicit_grace(self, name: str) -> None:
        job = RayClusterJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            extra_env={name: "7"},
        )
        assert job._build_env()[name] == "7"

    def test_external_code_libraries_custom_sglang_matches_runtime_env(self) -> None:
        job = RayClusterJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            engine="sglang_custom",
        )
        libs = job._external_code_libraries()
        assert libs == [
            str(job.verl_root / "sber_src" / "contribs" / "sglang-pretrain" / "python"),
            str(job.verl_root),
        ]

    def test_external_code_libraries_default_engine_returns_verl_root(self) -> None:
        job = RayClusterJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
        )
        assert job._external_code_libraries() == [str(job.verl_root)]


@pytest.mark.unit
class TestRayTrainJobVERL:
    def test_tb_stamp_uses_resolved_scalar_parent(self, monkeypatch) -> None:
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            script_path="/tmp/run.sh",
            runtime_env_config_path="/tmp/runtime.yaml",
        )
        monkeypatch.setattr(
            "job_submitter.jobs.verl.train.resolve_verl_logs_path",
            lambda *args, **kwargs: "/tensorboard/project/experiment",
        )
        monkeypatch.setattr(job, "_resolve_job_logdir", lambda: "/tmp/jobs")

        assert job._resolve_tb_log_dir() == "/tensorboard/project"

    def test_replace_runtime_env_config_adds_pam_envs_when_config_set(
        self, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "runtime_env.yaml"
        OmegaConf.save(
            {"excludes": ["/sber_src/contribs/pretrain_auto_metrics/"]}, cfg_path
        )

        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            script_path="/tmp/run.sh",
            runtime_env_config_path=str(cfg_path),
            pam_config_path="/tmp/pam.yaml",
        )
        job._replace_runtime_env_config(cfg_path, {"PAM_CONFIG_PATH": "/tmp/pam.yaml"})

        new_cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        assert "/sber_src/contribs/pretrain_auto_metrics/" in new_cfg["excludes"]
        assert new_cfg["env_vars"]["PAM_CONFIG_PATH"] == "/tmp/pam.yaml"
        assert cfg_path.with_suffix(".backup.yaml").exists()

    def test_build_env_adds_sharp_flag(self, monkeypatch, tmp_path: Path) -> None:
        cfg_path = tmp_path / "runtime_env.yaml"
        OmegaConf.save({"excludes": []}, cfg_path)
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            script_path="/tmp/run.sh",
            runtime_env_config_path=str(cfg_path),
            use_sharp=True,
        )
        monkeypatch.setattr(
            job, "_replace_runtime_env_config", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(job, "_resolve_slurm_jwt_token", lambda: "token")
        monkeypatch.setattr(job, "_resolve_slurmrestd_urls", lambda: ["http://a:6820"])
        env = job._build_env()
        assert env["NCCL_COLLNET_ENABLE"] == "1"

    @pytest.mark.parametrize("pam_config_path", [None, "/tmp/pam.yaml"])
    @pytest.mark.parametrize("existing_env", [None, {"EXISTING": "1"}])
    def test_replace_runtime_env_config_merges_wandb_envs_and_preserves_backup(
        self,
        tmp_path: Path,
        pam_config_path: str | None,
        existing_env: dict[str, str] | None,
    ) -> None:
        job = self._train_job(tmp_path, pam_config_path=pam_config_path)
        cfg_path = Path(job.runtime_env_config_path)
        original = {"working_dir": "./", "excludes": ["*.log"]}
        if existing_env is not None:
            original["env_vars"] = existing_env
        OmegaConf.save(original, cfg_path)
        original_bytes = cfg_path.read_bytes()
        wandb_envs = {
            "WANDB_BASE_URL": "https://wandb.example.com",
            "WANDB_INSECURE_DISABLE_SSL": "false",
            "WANDB_DIR": "/custom/wandb",
        }
        runtime_envs = {**wandb_envs, "PAM_CONFIG_PATH": "/tmp/pam.yaml"}

        job._replace_runtime_env_config(cfg_path, runtime_envs)

        expected_env = {**(existing_env or {}), **wandb_envs}
        expected_excludes = ["*.log"]
        if pam_config_path:
            expected_env["PAM_CONFIG_PATH"] = "/tmp/pam.yaml"
        else:
            expected_excludes.append("/sber_src/contribs/pretrain_auto_metrics/")
        assert OmegaConf.to_container(OmegaConf.load(cfg_path)) == {
            "working_dir": "./",
            "excludes": expected_excludes,
            "env_vars": expected_env,
        }
        assert cfg_path.with_suffix(".backup.yaml").read_bytes() == original_bytes

    @pytest.mark.parametrize("pam_config_path", [None, "/tmp/pam.yaml"])
    def test_replace_runtime_env_config_preserves_wandb_values_absent_from_source(
        self,
        tmp_path: Path,
        pam_config_path: str | None,
    ) -> None:
        job = self._train_job(tmp_path, pam_config_path=pam_config_path)
        cfg_path = Path(job.runtime_env_config_path)
        original_env = {
            "WANDB_BASE_URL": "https://wandb.example.com",
            "WANDB_INSECURE_DISABLE_SSL": "false",
            "WANDB_DIR": "/custom/wandb",
            "WANDB_API_KEY": "test-key",
        }
        OmegaConf.save({"env_vars": original_env}, cfg_path)

        job._replace_runtime_env_config(cfg_path, {})

        assert (
            OmegaConf.to_container(OmegaConf.load(cfg_path))["env_vars"] == original_env
        )

    @pytest.mark.parametrize("pam_config_path", [None, "/tmp/pam.yaml"])
    @pytest.mark.parametrize(
        ("region", "submission_type", "base_url", "disable_ssl", "wandb_dir"),
        [
            (
                "o1",
                "slurm",
                "https://wb.static.space",
                "true",
                "/mnt/nfs/07-share/checkpoints/wandb",
            ),
            (
                "o2",
                "slurm",
                "https://wb.static.space",
                "true",
                "/mnt/nfs/07-share/checkpoints/wandb",
            ),
            (
                "o5",
                "slurm",
                "https://wb.static.space",
                "true",
                "/mnt/nfs/07-share/checkpoints/wandb",
            ),
            (
                "SR004",
                "cloud",
                "https://wandb.sberdevices.ru",
                "true",
                "/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts",
            ),
            (
                "SR006",
                "cloud",
                "https://wandb.sberdevices.ru",
                None,
                "/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts",
            ),
            (
                "SR008",
                "cloud",
                "https://wandb.sberdevices.ru",
                None,
                "/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts",
            ),
        ],
    )
    def test_build_env_propagates_region_wandb_settings_to_runtime_yaml(
        self,
        tmp_path: Path,
        pam_config_path: str | None,
        region: str,
        submission_type: str,
        base_url: str,
        disable_ssl: str | None,
        wandb_dir: str,
    ) -> None:
        job = self._train_job(
            tmp_path,
            region=region,
            submission_type=submission_type,
            pam_config_path=pam_config_path,
        )
        OmegaConf.save(
            {
                "env_vars": {
                    "WANDB_BASE_URL": "https://old.example.com",
                    "WANDB_DIR": "/old/wandb",
                }
            },
            job.runtime_env_config_path,
        )

        env = job._build_env()

        runtime_env = OmegaConf.to_container(
            OmegaConf.load(job.runtime_env_config_path)
        )["env_vars"]
        for actual in (env, runtime_env):
            assert actual["WANDB_BASE_URL"] == base_url
            assert actual["WANDB_DIR"] == wandb_dir
            assert actual.get("WANDB_INSECURE_DISABLE_SSL") == disable_ssl

    @pytest.mark.parametrize("pam_config_path", [None, "/tmp/pam.yaml"])
    def test_build_env_wandb_extra_env_overrides_region_and_runtime_yaml(
        self,
        tmp_path: Path,
        pam_config_path: str | None,
    ) -> None:
        overrides = {
            "WANDB_BASE_URL": "https://wandb.example.com",
            "WANDB_INSECURE_DISABLE_SSL": "false",
            "WANDB_DIR": "/custom/wandb",
        }
        job = self._train_job(
            tmp_path,
            region="o1",
            submission_type="slurm",
            pam_config_path=pam_config_path,
            extra_env=overrides,
        )
        OmegaConf.save(
            {
                "env_vars": {
                    "WANDB_BASE_URL": "https://old.example.com",
                    "WANDB_INSECURE_DISABLE_SSL": "true",
                    "WANDB_DIR": "/old/wandb",
                    "WANDB_API_KEY": "test-key",
                }
            },
            job.runtime_env_config_path,
        )

        env = job._build_env()

        runtime_env = OmegaConf.to_container(
            OmegaConf.load(job.runtime_env_config_path)
        )["env_vars"]
        for key, value in overrides.items():
            assert env[key] == value
            assert runtime_env[key] == value
        assert runtime_env["WANDB_API_KEY"] == "test-key"

    def test_build_env_adds_pam_base_image_when_set(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "runtime_env.yaml"
        OmegaConf.save({"excludes": []}, cfg_path)
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            script_path="/tmp/run.sh",
            runtime_env_config_path=str(cfg_path),
            pam_config_path="/tmp/pam.yaml",
            pam_base_image="/tmp/images/custom_pam.sqsh",
        )
        monkeypatch.setattr(
            job, "_replace_runtime_env_config", lambda *args, **kwargs: None
        )
        env = job._build_env()
        assert env["PAM_BASE_IMAGE"] == "/tmp/images/custom_pam.sqsh"

    def test_build_env_adds_pam_queue_name_to_runtime_env(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "runtime_env.yaml"
        OmegaConf.save({"excludes": []}, cfg_path)
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            script_path="/tmp/run.sh",
            runtime_env_config_path=str(cfg_path),
            pam_config_path="/tmp/pam.yaml",
            pam_queue_name="pretrain-ml",
        )

        env = job._build_env()
        updated = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        assert env["PAM_QUEUE_NAME"] == "pretrain-ml"
        assert updated["env_vars"]["PAM_QUEUE_NAME"] == "pretrain-ml"

    def test_build_env_omits_pam_base_image_when_unset(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "runtime_env.yaml"
        OmegaConf.save({"excludes": []}, cfg_path)
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            script_path="/tmp/run.sh",
            runtime_env_config_path=str(cfg_path),
            pam_config_path="/tmp/pam.yaml",
        )
        monkeypatch.setattr(
            job, "_replace_runtime_env_config", lambda *args, **kwargs: None
        )
        env = job._build_env()
        assert "PAM_BASE_IMAGE" not in env

    def _train_job(self, tmp_path: Path, **overrides):
        cfg_path = tmp_path / "runtime_env.yaml"
        OmegaConf.save({"excludes": []}, cfg_path)
        script_path = tmp_path / "run.sh"
        script_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        params = dict(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            script_path=str(script_path),
            runtime_env_config_path=str(cfg_path),
        )
        params.update(overrides)
        return RayTrainJobVERL(**params)

    def test_build_env_exports_js_engine_and_js_base_image(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        # A job submitted by this job (the verl Megatron->HF converter) reads these to run
        # on the same engine build and the same image the training job runs on.
        job = self._train_job(tmp_path, engine="sglang_custom")
        monkeypatch.setattr(
            job, "_replace_runtime_env_config", lambda *args, **kwargs: None
        )
        env = job._build_env()

        assert env["JS_ENGINE"] == "sglang_custom"
        assert env["JS_BASE_IMAGE"] == VERL_BASE_IMAGES["sglang_custom"]["SR008"]

    def test_build_env_js_base_image_follows_an_explicit_override(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        job = self._train_job(tmp_path, base_image="/tmp/images/pinned.sqsh")
        monkeypatch.setattr(
            job, "_replace_runtime_env_config", lambda *args, **kwargs: None
        )
        assert job._build_env()["JS_BASE_IMAGE"] == "/tmp/images/pinned.sqsh"

    def test_build_env_exports_pam_config_path_when_set(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        job = self._train_job(tmp_path, pam_config_path="/tmp/pam.yaml")
        monkeypatch.setattr(
            job, "_replace_runtime_env_config", lambda *args, **kwargs: None
        )
        assert job._build_env()["PAM_CONFIG_PATH"] == "/tmp/pam.yaml"

    def test_build_env_omits_pam_config_path_when_pam_is_disabled(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        # A None here used to be exported verbatim and stringified to "None" downstream --
        # a truthy path, so consumers fired PAM against a config that does not exist.
        job = self._train_job(tmp_path, pam_config_path=None)
        monkeypatch.setattr(
            job, "_replace_runtime_env_config", lambda *args, **kwargs: None
        )
        env = job._build_env()

        assert "PAM_CONFIG_PATH" not in env
        assert None not in env.values()

    def test_build_env_omits_pam_config_path_for_the_literal_none_string(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        job = self._train_job(tmp_path, pam_config_path="None")
        monkeypatch.setattr(
            job, "_replace_runtime_env_config", lambda *args, **kwargs: None
        )
        assert "PAM_CONFIG_PATH" not in job._build_env()

    @pytest.mark.parametrize("engine", ["sglang", "sglang_custom"])
    def test_o2_has_a_verl_base_image(self, engine: str) -> None:
        # o2 is a supported region (_resolve_turn_off_first_srun branches on it), but the
        # verl registry used to carry only o1/o5, so _resolve_base_image raised a KeyError.
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o2",
            engine=engine,
            script_path="/tmp/run.sh",
            runtime_env_config_path="/tmp/runtime.yaml",
        )
        image = job._resolve_base_image()

        assert image == VERL_BASE_IMAGES[engine]["o2"]
        assert image.startswith("/mnt/nfs/07-share/")
        assert image.endswith(".sqsh")

    def test_script_slurm_branch_and_before_submit(self) -> None:
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            script_path="/tmp/run.sh",
            runtime_env_config_path="/tmp/runtime.yaml",
        )
        assert "setup_ray_cluster/slurm.sh" in job._script()
        assert job._before_submit() is None

    def test_tracking_paths_are_explicit(self) -> None:
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            script_path="/tmp/run.sh",
            runtime_env_config_path="/tmp/runtime.yaml",
            pam_config_path="/tmp/pam.yaml",
        )

        assert job._tracking_config_path(None) == "/tmp/pam.yaml"
        assert job._tracking_script_path(None) == "/tmp/run.sh"

    def test_replace_runtime_env_config_without_pam_adds_exclude(
        self, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "runtime_env.yaml"
        OmegaConf.save({}, cfg_path)
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            script_path="/tmp/run.sh",
            runtime_env_config_path=str(cfg_path),
            pam_config_path=None,
        )
        job._replace_runtime_env_config(cfg_path, {})
        loaded = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        assert "/sber_src/contribs/pretrain_auto_metrics/" in loaded["excludes"]

    def test_replace_runtime_env_config_with_existing_env_vars_keeps_exclude_when_missing(
        self, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "runtime_env.yaml"
        OmegaConf.save({"excludes": [], "env_vars": {"EXISTING": "1"}}, cfg_path)

        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            script_path="/tmp/run.sh",
            runtime_env_config_path=str(cfg_path),
            pam_config_path="/tmp/pam.yaml",
        )
        job._replace_runtime_env_config(cfg_path, {"PAM_CONFIG_PATH": "/tmp/pam.yaml"})

        updated = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        assert updated["excludes"] == []
        assert updated["env_vars"]["EXISTING"] == "1"
        assert updated["env_vars"]["PAM_CONFIG_PATH"] == "/tmp/pam.yaml"

    def test_replace_runtime_env_config_without_pam_does_not_duplicate_exclude(
        self, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "runtime_env.yaml"
        OmegaConf.save(
            {"excludes": ["/sber_src/contribs/pretrain_auto_metrics/"]}, cfg_path
        )

        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            script_path="/tmp/run.sh",
            runtime_env_config_path=str(cfg_path),
            pam_config_path=None,
        )
        job._replace_runtime_env_config(cfg_path, {})

        updated = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        assert updated["excludes"] == ["/sber_src/contribs/pretrain_auto_metrics/"]

    @pytest.mark.parametrize(
        ("region", "explicit_value", "expected"),
        [
            ("o1", None, True),
            ("o2", None, True),
            ("o5", None, False),
            ("o1", False, False),
            ("o5", True, True),
        ],
    )
    def test_resolve_turn_off_first_srun(
        self, region: str, explicit_value: bool | None, expected: bool
    ) -> None:
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region=region,
            script_path="/tmp/run.sh",
            runtime_env_config_path="/tmp/runtime.yaml",
            turn_off_first_srun=explicit_value,
        )
        assert job._resolve_turn_off_first_srun() is expected

    def test_submit_calls_anonimize_config_then_submits_cfg(self, monkeypatch) -> None:
        called = {}
        fake_cfg = object()
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            script_path="/tmp/run.sh",
            runtime_env_config_path="/tmp/runtime.yaml",
            pam_config_path="/tmp/pam.yaml",
        )
        monkeypatch.setattr(
            job, "_anonimize_config", lambda p: called.setdefault("anonimize", p)
        )
        monkeypatch.setattr(job, "to_config", lambda: fake_cfg)
        monkeypatch.setattr(job, "assert_job", lambda job_obj, cfg: None)
        monkeypatch.setattr(job, "_before_submit", lambda: None)
        monkeypatch.setattr(job, "_submit_config", lambda cfg: "ok")

        assert job.submit() == "ok"
        assert called["anonimize"] == "/tmp/pam.yaml"

    def test_submit_dry_run_calls_before_submit_and_skips_backend(
        self, monkeypatch
    ) -> None:
        job = RayTrainJobVERL(
            user_login="u",
            exp_name="e",
            submission_type="slurm",
            region="o1",
            script_path="/tmp/run.sh",
            runtime_env_config_path="/tmp/runtime.yaml",
            dry_run=True,
        )
        assert_dry_run_submit_contract(
            monkeypatch=monkeypatch,
            job=job,
            patch_anonimize=True,
        )
