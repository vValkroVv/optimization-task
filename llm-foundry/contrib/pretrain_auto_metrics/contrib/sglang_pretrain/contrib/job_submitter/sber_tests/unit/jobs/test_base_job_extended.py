from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from sber_tests.dry_run_helpers import assert_dry_run_submit_contract
from sber_tests.factories import DummyJob


def make_cloud_job(region: str = "SR008", **kwargs) -> DummyJob:
    defaults = {
        "user_login": "u",
        "exp_name": "e",
        "region": region,
        "submission_type": "cloud",
    }
    defaults.update(kwargs)
    return DummyJob(**defaults)


def make_slurm_job(region: str = "o1", **kwargs) -> DummyJob:
    defaults = {
        "user_login": "u",
        "exp_name": "e",
        "region": region,
        "submission_type": "slurm",
    }
    defaults.update(kwargs)
    return DummyJob(**defaults)


@pytest.mark.unit
class TestBaseJobSubprocessAndRegion:
    def test_run_subprocess_success(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "job_submitter.jobs.base.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout="ok\n", stderr=""
            ),
        )
        assert DummyJob._run_subprocess("echo ok") == "ok"

    def test_run_subprocess_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "job_submitter.jobs.base.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=1, stdout="", stderr="boom"
            ),
        )
        with pytest.raises(ValueError):
            DummyJob._run_subprocess("bad")

    def test_build_env_stamps_tensorboard_path_and_comment(self, monkeypatch) -> None:
        monkeypatch.setenv("SLURM_JWT_TOKEN", "jwt")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820")
        job = make_slurm_job(
            queue_name="pretrain", extra_env={"TB_LOG_DIR": "/data/tensorboard/run"}
        )

        config = job.to_config()

        assert config.env_variables["TB_LOG_DIR"] == "/data/tensorboard/run"
        assert config.comment == "/data/tensorboard/run"

    def test_build_env_keeps_submit_working_when_tb_resolution_fails(
        self, monkeypatch, caplog
    ) -> None:
        job = make_slurm_job()
        monkeypatch.setattr(
            job,
            "_resolve_tb_log_dir",
            lambda: (_ for _ in ()).throw(ValueError("broken config")),
        )

        env = job._build_env()

        assert "TB_LOG_DIR" not in env
        assert "non-blocking" in caplog.text

    def test_get_current_region_falls_back_to_slurm_cluster_name(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.delenv("CLUSTER_KEY", raising=False)
        job = make_cloud_job()
        monkeypatch.setattr(job, "_get_slurm_cluster_name", lambda: "o1")
        assert job._get_current_region() == ("o1", "slurm")

    def test_get_current_region_prefers_slurm_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "o5")
        monkeypatch.setenv("CLUSTER_KEY", "SR008")
        job = make_cloud_job()
        assert job._get_current_region() == ("o5", "slurm")

    def test_resolve_region_prefers_explicit_region_over_slurm_env(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "o5")
        job = make_slurm_job(region="o1")
        assert job._resolve_region() == ("o1", "slurm")

    def test_resolve_region_normalizes_cyrillic_o(self, monkeypatch) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "o5")
        job = make_slurm_job(region="\u043e1")
        assert job._resolve_region() == ("o1", "slurm")

    def test_get_current_region_normalizes_cyrillic_o_in_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "\u043e5")
        monkeypatch.delenv("CLUSTER_KEY", raising=False)
        job = make_cloud_job(region=None)
        assert job._get_current_region() == ("o5", "slurm")

    def test_normalize_region_blank_and_unknown_values(self) -> None:
        assert DummyJob._normalize_region("   ") is None
        assert DummyJob._normalize_region("custom-region") == "custom-region"

    def test_normalize_submission_type_none(self) -> None:
        assert DummyJob._normalize_submission_type(None) is None

    def test_infer_submission_type_for_region_all_branches(self) -> None:
        assert DummyJob._infer_submission_type_for_region("o1") == "slurm"
        assert DummyJob._infer_submission_type_for_region("SR008") == "cloud"
        assert DummyJob._infer_submission_type_for_region("custom-region") is None

    def test_get_current_region_uses_cloud_env_when_slurm_env_missing(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "sr008")
        job = make_cloud_job(region=None)
        monkeypatch.setattr(job, "_get_slurm_cluster_name", lambda: "o1")
        assert job._get_current_region() == ("SR008", "cloud")

    def test_resolve_region_without_explicit_region_uses_detected_context(
        self, monkeypatch
    ) -> None:
        job = make_cloud_job(region=None, submission_type=None)
        monkeypatch.setattr(job, "_get_current_region", lambda: ("SR008", "cloud"))
        assert job._resolve_region() == ("SR008", "cloud")

    def test_parse_cluster_name_raises_when_not_found(self) -> None:
        with pytest.raises(ValueError):
            DummyJob._parse_cluster_name("no cluster key")

    def test_get_slurm_cluster_name_uses_run_and_parse(self, monkeypatch) -> None:
        job = make_slurm_job()
        monkeypatch.setattr(
            job, "_run_subprocess", lambda cmd: "ClusterName = my_cluster"
        )
        assert job._get_slurm_cluster_name() == "mycluster"


@pytest.mark.unit
class TestBaseJobSlurmResolvers:
    def test_resolve_slurm_jwt_token_from_env(self, monkeypatch) -> None:
        job = make_slurm_job()
        monkeypatch.setenv("SLURM_JWT_TOKEN", "env-token")
        assert job._resolve_slurm_jwt_token() == "env-token"

    def test_resolve_slurm_jwt_token_from_subprocess(self, monkeypatch) -> None:
        job = make_slurm_job()
        monkeypatch.delenv("SLURM_JWT_TOKEN", raising=False)
        monkeypatch.setattr(job, "_run_subprocess", lambda cmd: "SLURM_JWT=generated")
        assert job._resolve_slurm_jwt_token() == "generated"

    def test_resolve_slurmrestd_urls_from_env(self, monkeypatch) -> None:
        job = make_slurm_job()
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820||http://b:6820")
        assert job._resolve_slurmrestd_urls() == ["http://a:6820", "http://b:6820"]

    def test_resolve_slurmrestd_urls_from_subprocess(self, monkeypatch) -> None:
        job = make_slurm_job()
        monkeypatch.delenv("SLURMRESTD_URLS", raising=False)
        monkeypatch.setattr(
            job,
            "_run_subprocess",
            lambda cmd: "SlurmctldHost[0]=h1\nSlurmctldHost[1]=h2\n",
        )
        assert job._resolve_slurmrestd_urls() == ["http://h1:6820", "http://h2:6820"]

    def test_resolve_job_and_slurm_logdir_defaults(self) -> None:
        job = make_slurm_job()
        assert job._resolve_job_logdir().endswith("jobs_logs")
        assert job._resolve_slurm_logdir().endswith("slurm_logs")

    def test_build_env_slurm_skips_cluster_name_when_env_var_missing(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("SLURM_JWT_TOKEN", "env-token")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820")
        job = make_slurm_job()
        env = job._build_env()
        assert "SLURM_CLUSTER_NAME" not in env

    def test_build_env_cloud_sets_hf_offline_when_internet_disabled(self) -> None:
        job = make_cloud_job(internet=False)
        env = job._build_env()
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["HF_DATASETS_OFFLINE"] == "1"

    def test_build_env_exports_js_base_image(self) -> None:
        # A job this job submits in turn can reuse the exact image it runs on.
        job = make_cloud_job(base_image="/tmp/images/pinned.sqsh")
        assert job._build_env()["JS_BASE_IMAGE"] == "/tmp/images/pinned.sqsh"

    def test_build_env_omits_js_base_image_when_resolution_fails(
        self, monkeypatch
    ) -> None:
        # Best-effort: subclasses assert on the engine/region pair inside
        # _resolve_base_image, and building the env must not be where that surfaces.
        job = make_cloud_job()

        def _boom() -> str:
            raise AssertionError("unsupported engine/region pair")

        monkeypatch.setattr(job, "_resolve_base_image", _boom)

        assert "JS_BASE_IMAGE" not in job._build_env()

    def test_build_env_omits_js_base_image_when_unresolved(self, monkeypatch) -> None:
        job = make_cloud_job()
        monkeypatch.setattr(job, "_resolve_base_image", lambda: None)

        assert "JS_BASE_IMAGE" not in job._build_env()

    @pytest.mark.parametrize(
        ("job_kwargs", "expected_workspace"),
        [
            (
                {
                    "workspace_name": " explicit-workspace ",
                    "extra_env": {"MLS_NTB_WORKSPACE_NAME": "extra-workspace"},
                },
                "explicit-workspace",
            ),
            (
                {"extra_env": {"MLS_NTB_WORKSPACE_NAME": " extra-workspace "}},
                "extra-workspace",
            ),
            ({}, "environment-workspace"),
        ],
    )
    def test_build_env_cloud_resolves_workspace(
        self, monkeypatch, job_kwargs, expected_workspace
    ) -> None:
        monkeypatch.setenv("MLS_NTB_WORKSPACE_NAME", "environment-workspace")
        job = make_cloud_job(**job_kwargs)

        cfg = job.to_config()

        assert cfg.env_variables["MLS_NTB_WORKSPACE_NAME"] == expected_workspace

    @pytest.mark.parametrize(
        "job_kwargs",
        [
            {},
            {"workspace_name": " "},
            {"extra_env": {"MLS_NTB_WORKSPACE_NAME": " "}},
        ],
    )
    def test_build_env_cloud_rejects_missing_workspace(
        self, monkeypatch, job_kwargs
    ) -> None:
        monkeypatch.delenv("MLS_NTB_WORKSPACE_NAME", raising=False)
        job = make_cloud_job(**job_kwargs)

        with pytest.raises(ValueError, match="Cloud workspace is not specified"):
            job.to_config()

    def test_build_env_slurm_does_not_require_workspace(self, monkeypatch) -> None:
        monkeypatch.delenv("MLS_NTB_WORKSPACE_NAME", raising=False)
        monkeypatch.setenv("SLURM_JWT_TOKEN", "env-token")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820")
        job = make_slurm_job(queue_name="pretrain")

        cfg = job.to_config()

        assert "MLS_NTB_WORKSPACE_NAME" not in cfg.env_variables

    def test_resolve_partition_processes_and_allocation_helpers(self) -> None:
        slurm_job = make_slurm_job(region="o5", priority_class="high")
        assert slurm_job._resolve_partition() == "main"
        assert slurm_job._resolve_processes_per_worker() == 1
        assert slurm_job._resolve_allocation_name() is None

        leads_job = make_slurm_job(priority_class="leads")
        assert leads_job._resolve_partition() == "leads"

        cloud_job = make_cloud_job()
        assert (
            cloud_job._resolve_allocation_name()
            == "alloc-sberdevices-giga-vision-h100-sr008"
        )

        giga_audio_job = make_cloud_job(queue_name="giga-audio")
        assert (
            giga_audio_job._resolve_allocation_name() == "alloc-giga-audio-h100-sr008"
        )

    @pytest.mark.parametrize(
        ("priority_class", "expected"),
        [
            ("urgent", ("data", "data_high", "data")),
            ("high", ("data", "data", "data")),
            ("medium", ("data", "data_low", "data")),
            ("low", ("main", "normal", "data")),
            ("leads", ("leads", "leads", "leads")),
        ],
    )
    @pytest.mark.parametrize("region", ["o1", "o2"])
    def test_team_qos_priority_class_maps_to_team_partition_qos_and_account(
        self, region, priority_class: str, expected: tuple[str, str, str]
    ) -> None:
        job = make_slurm_job(
            region=region, queue_name="data", priority_class=priority_class
        )

        resolved = (
            job._resolve_partition(),
            job._resolve_slurm_qos(),
            job._resolve_slurm_account(),
        )

        assert resolved == expected

    @pytest.mark.parametrize("region", ["o1", "o2"])
    def test_team_qos_unknown_priority_class_resolves_no_qos(self, region) -> None:
        job = make_slurm_job(region=region, queue_name="data", priority_class="weird")
        assert job._resolve_slurm_qos() is None

    @pytest.mark.parametrize("region", ["o1", "o2"])
    def test_team_qos_defaults_to_slurm_default_account(
        self, region, monkeypatch
    ) -> None:
        commands: list[str] = []

        def _run(command: str) -> str:
            commands.append(command)
            return "pretrain"

        monkeypatch.setattr(DummyJob, "_run_subprocess", staticmethod(_run))
        monkeypatch.setattr("job_submitter.jobs.base.getpass.getuser", lambda: "me")
        job = make_slurm_job(region=region, priority_class="high")

        assert job._resolve_slurm_qos() == "pretrain"
        assert job._resolve_partition() == "pretrain"
        assert job.queue_name == "pretrain"
        assert commands == ["sacctmgr -nP show user me format=defaultaccount"]

    def test_legacy_account_follows_priority_class(self) -> None:
        high = make_slurm_job(region="o5", priority_class="high", queue_name="pretrain")
        assert high._resolve_slurm_account() == "pretrain"
        assert high._resolve_slurm_qos() == "pretrain"
        assert (
            make_slurm_job(
                region="o5", priority_class="medium"
            )._resolve_slurm_account()
            is None
        )
        assert (
            make_slurm_job(region="o5", priority_class="leads")._resolve_slurm_account()
            == "leads"
        )
        assert make_cloud_job()._resolve_slurm_account() is None


@pytest.mark.unit
class TestBaseJobConfigReplacementAndSubmit:
    def test_anonimize_config_updates_slurm_autometrics_fields(
        self, tmp_path: Path
    ) -> None:
        chat = tmp_path / "chat.jinja"
        chat.write_text("plain text", encoding="utf-8")
        cfg_path = tmp_path / "pam.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "obm_proxy": {
                        "chat-template": str(chat),
                        "chat-template-kwargs": {"default_devsystem": "not-base64"},
                    },
                    "evaluators": {
                        "obm": {
                            "all": {"groups": ["sber-team"]},
                        }
                    },
                }
            },
            cfg_path,
        )

        job = make_slurm_job()
        job._anonimize_config(cfg_path)

        updated = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        assert updated["autometrics"]["obm_proxy"]["chat-template"].endswith(".base64")
        assert (
            updated["autometrics"]["obm_proxy"]["chat-template-kwargs"][
                "default_devsystem"
            ]
            != "not-base64"
        )
        assert updated["autometrics"]["evaluators"]["obm"]["all"]["groups"] == [
            "sber-team"
        ]
        assert not cfg_path.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_skips_for_non_slurm_submission(
        self, tmp_path: Path
    ) -> None:
        non_slurm = make_cloud_job()
        cfg = tmp_path / "cfg.yaml"
        OmegaConf.save({"x": 1}, cfg)
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        non_slurm._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before
        assert not cfg.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_skips_when_config_path_is_none(self) -> None:
        slurm_job = make_slurm_job()
        slurm_job._anonimize_config(None)

    def test_anonimize_config_skips_without_autometrics_section(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        cfg = tmp_path / "cfg.yaml"
        OmegaConf.save({"x": 1}, cfg)
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        slurm_job._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before
        assert not cfg.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_skips_without_inference_section(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        cfg2 = tmp_path / "cfg2.yaml"
        OmegaConf.save({"autometrics": {}}, cfg2)
        before = OmegaConf.to_container(OmegaConf.load(cfg2), resolve=True)

        slurm_job._anonimize_config(cfg2)

        after = OmegaConf.to_container(OmegaConf.load(cfg2), resolve=True)
        assert after == before
        assert not cfg2.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_skips_without_chat_template(self, tmp_path: Path) -> None:
        slurm_job = make_slurm_job()
        cfg3 = tmp_path / "cfg3.yaml"
        OmegaConf.save({"autometrics": {"inference": {"x": 1}}}, cfg3)
        before = OmegaConf.to_container(OmegaConf.load(cfg3), resolve=True)

        slurm_job._anonimize_config(cfg3)

        after = OmegaConf.to_container(OmegaConf.load(cfg3), resolve=True)
        assert after == before
        assert not cfg3.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_warns_and_returns_when_obm_proxy_template_missing(
        self, tmp_path: Path, capsys
    ) -> None:
        slurm_job = make_slurm_job()
        missing_template = tmp_path / "missing_chat_template.jinja"
        cfg = tmp_path / "cfg_missing_obm_proxy_template.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "obm_proxy": {"chat-template": str(missing_template)},
                    "evaluators": {"obm": {"all": {"groups": ["sber"]}}},
                }
            },
            cfg,
        )
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        slurm_job._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before
        assert (
            f"[WARNING] OBM proxy chat template {missing_template} not found, skipping"
            in capsys.readouterr().out
        )

    def test_anonimize_config_leaves_file_when_only_chat_template_exists(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        chat = tmp_path / "chat.jinja"
        chat.write_text("already base64", encoding="utf-8")
        cfg4 = tmp_path / "cfg4.yaml"
        OmegaConf.save(
            {"autometrics": {"inference": {"chat-template": str(chat)}, "other": 1}},
            cfg4,
        )
        before = OmegaConf.to_container(OmegaConf.load(cfg4), resolve=True)

        slurm_job._anonimize_config(cfg4)

        after = OmegaConf.to_container(OmegaConf.load(cfg4), resolve=True)
        assert after == before
        assert not cfg4.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_keeps_evaluators_when_no_optional_fields(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        chat = tmp_path / "chat.jinja"
        chat.write_bytes(b"cGxhaW4gdGV4dA==")
        cfg = tmp_path / "cfg_no_optional.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "inference": {"chat-template": str(chat)},
                    "evaluators": {"obm": {"all": {}}},
                }
            },
            cfg,
        )
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        slurm_job._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before
        assert not cfg.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_skips_encoding_for_already_base64_default_devsystem(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        chat = tmp_path / "chat2.jinja"
        chat.write_bytes(b"cGxhaW4gdGV4dA==")
        cfg = tmp_path / "cfg_encoded_devsystem.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "inference": {"chat-template": str(chat)},
                    "evaluators": {
                        "obm": {
                            "obm_proxy": {
                                "chat-template-kwargs": {
                                    "default_devsystem": "bm90LWJhc2U2NA=="
                                }
                            }
                        }
                    },
                }
            },
            cfg,
        )
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        slurm_job._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before
        assert not cfg.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_skips_group_rewrite_without_sber_marker(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        chat = tmp_path / "chat3.jinja"
        chat.write_bytes(b"cGxhaW4gdGV4dA==")
        cfg = tmp_path / "cfg_group_same.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "inference": {"chat-template": str(chat)},
                    "evaluators": {"obm": {"all": {"groups": ["papaya-team"]}}},
                }
            },
            cfg,
        )
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        slurm_job._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before
        assert not cfg.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_ignores_legacy_group_field(self, tmp_path: Path) -> None:
        slurm_job = make_slurm_job()
        chat = tmp_path / "chat4.jinja"
        chat.write_bytes(b"cGxhaW4gdGV4dA==")
        cfg = tmp_path / "cfg_group_legacy.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "inference": {"chat-template": str(chat)},
                    "evaluators": {"obm": {"all": {"group": "SBER"}}},
                }
            },
            cfg,
        )
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        slurm_job._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before
        assert not cfg.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_rewrites_exact_sber_in_groups_list(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        chat = tmp_path / "chat5.jinja"
        chat.write_bytes(b"cGxhaW4gdGV4dA==")
        cfg = tmp_path / "cfg_group_list_exact.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "inference": {"chat-template": str(chat)},
                    "evaluators": {"obm": {"all": {"groups": ["assistant", "sber"]}}},
                }
            },
            cfg,
        )

        slurm_job._anonimize_config(cfg)

        updated = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert updated["autometrics"]["evaluators"]["obm"]["all"]["groups"] == [
            "assistant",
            "papaya",
        ]
        assert not cfg.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_skips_when_all_is_not_mapping(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        chat = tmp_path / "chat6.jinja"
        chat.write_bytes(b"cGxhaW4gdGV4dA==")
        cfg = tmp_path / "cfg_all_not_mapping.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "inference": {"chat-template": str(chat)},
                    "evaluators": {"obm": {"all": ["sber"]}},
                }
            },
            cfg,
        )
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        slurm_job._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before
        assert not cfg.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_legacy_group_without_sber_kept(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        chat = tmp_path / "chat7.jinja"
        chat.write_bytes(b"cGxhaW4gdGV4dA==")
        cfg = tmp_path / "cfg_legacy_group_kept.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "inference": {"chat-template": str(chat)},
                    "evaluators": {"obm": {"all": {"group": "PAPAYA"}}},
                }
            },
            cfg,
        )
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        slurm_job._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before
        assert not cfg.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_groups_list_with_non_str_items(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        chat = tmp_path / "chat8.jinja"
        chat.write_bytes(b"cGxhaW4gdGV4dA==")
        cfg = tmp_path / "cfg_groups_mixed.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "inference": {"chat-template": str(chat)},
                    "evaluators": {"obm": {"all": {"groups": ["sber", 1]}}},
                }
            },
            cfg,
        )

        slurm_job._anonimize_config(cfg)

        updated = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert updated["autometrics"]["evaluators"]["obm"]["all"]["groups"] == [
            "papaya",
            1,
        ]
        assert not cfg.with_suffix(".backup.yaml").exists()

    def test_anonimize_config_obm_proxy_uses_snake_case_chat_template_key(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        chat = tmp_path / "chat_snake.jinja"
        chat.write_text("plain text", encoding="utf-8")
        cfg = tmp_path / "cfg_snake_case_chat_template.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "obm_proxy": {"chat_template": str(chat)},
                    "evaluators": [],
                }
            },
            cfg,
        )

        slurm_job._anonimize_config(cfg)

        updated = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert updated["autometrics"]["obm_proxy"]["chat_template"].endswith(".base64")

    def test_anonimize_config_obm_proxy_chat_template_already_base64_no_rewrite(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        chat = tmp_path / "chat_base64.jinja"
        chat.write_bytes(b"cGxhaW4gdGV4dA==")
        cfg = tmp_path / "cfg_base64_chat_template.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "obm_proxy": {"chat-template": str(chat)},
                    "evaluators": [],
                }
            },
            cfg,
        )
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        slurm_job._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before

    def test_anonimize_config_skips_non_dict_chat_template_kwargs(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        cfg = tmp_path / "cfg_non_dict_chat_kwargs.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "obm_proxy": {"chat-template-kwargs": "bad-type"},
                    "evaluators": [],
                }
            },
            cfg,
        )
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        slurm_job._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before

    def test_anonimize_config_skips_non_dict_obm_proxy_and_non_dict_evaluators(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        cfg = tmp_path / "cfg_non_dict_obm_proxy.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "obm_proxy": ["not", "dict"],
                    "evaluators": ["not", "dict"],
                }
            },
            cfg,
        )
        before = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

        slurm_job._anonimize_config(cfg)

        after = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
        assert after == before

    def test_resolve_instance_type_cloud_uses_explicit_value(self) -> None:
        cloud = make_cloud_job()
        cloud.instance_type = "custom"
        assert cloud._resolve_instance_type() == "custom"

    def test_resolve_instance_type_slurm_returns_explicit_or_none(self) -> None:
        slurm = make_slurm_job()
        assert slurm._resolve_instance_type() is None
        slurm.instance_type = "slurm-custom"
        assert slurm._resolve_instance_type() == "slurm-custom"

    def test_resolve_processes_per_worker_forces_one_for_slurm(self) -> None:
        slurm = make_slurm_job()
        slurm.processes_per_worker = 3
        assert slurm._resolve_processes_per_worker() == 1

    def test_resolve_priority_class_keeps_default_without_env(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("PAM_PRIORITY_CLASS", raising=False)
        slurm = make_slurm_job()
        assert slurm._resolve_priority_class() == "medium"

    def test_resolve_queue_name_prefers_explicit_value(self) -> None:
        queue_job = make_cloud_job(region="SR006")
        queue_job.queue_name = "manual"
        assert queue_job._resolve_queue_name() == "manual"

    def test_resolve_priority_class_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("PAM_PRIORITY_CLASS", "high")
        job = make_cloud_job()
        assert job._resolve_priority_class() == "high"

    def test_resolve_priority_class_prefers_explicit_value_over_env(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("PAM_PRIORITY_CLASS", "high")
        job = make_cloud_job(priority_class="low")
        assert job._resolve_priority_class() == "low"

    def test_resolve_priority_class_dropped_on_shared_queue_despite_explicit_value(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("PAM_PRIORITY_CLASS", raising=False)
        job = make_cloud_job(priority_class="low", queue_name="shared")
        assert job._resolve_priority_class() is None

    def test_resolve_priority_class_from_env_on_shared_queue(self, monkeypatch) -> None:
        monkeypatch.setenv("PAM_PRIORITY_CLASS", "high")
        job = make_cloud_job(queue_name="shared")
        assert job._resolve_priority_class() == "high"

    def test_resolve_priority_class_drops_default_on_shared_queue(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("PAM_PRIORITY_CLASS", raising=False)
        job = make_cloud_job(queue_name="shared")
        assert job._resolve_priority_class() is None

    @pytest.mark.parametrize("value", [True, False])
    def test_resolve_turn_off_first_srun_prefers_explicit_value(
        self, value: bool
    ) -> None:
        job = make_slurm_job(turn_off_first_srun=value)
        assert job._resolve_turn_off_first_srun() is value

    @pytest.mark.parametrize("region", ["SR004", "SR006"])
    def test_queue_name_defaults_for_regions_without_default_queue(
        self, monkeypatch, region: str
    ) -> None:
        monkeypatch.delenv("PAM_QUEUE_NAME", raising=False)
        job = make_cloud_job(region=region)
        assert job._resolve_queue_name() is None

    def test_queue_name_defaults_to_none_for_other_regions_without_env(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("PAM_QUEUE_NAME", raising=False)
        job = make_cloud_job(region="o1")
        assert job._resolve_queue_name() is None

    def test_submit_calls_backend(self, monkeypatch) -> None:
        captured = {}

        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")
        job = make_cloud_job()
        submitted = SimpleNamespace(
            job_name="cloud-job-1",
            status=lambda: "Job status=Completed",
            logs=lambda **_: None,
        )

        def _fake_submit(self, cfg):
            captured["cfg"] = cfg
            self.last_submitted_job = submitted

        monkeypatch.setattr("job_submitter.jobs.base.CloudBackend.submit", _fake_submit)
        assert job.submit() is job
        assert captured["cfg"].submission_type == "cloud"
        assert captured["cfg"].job_desc == "u | e"
        assert job._last_submitted_job is submitted

    def test_submit_dry_run_calls_before_submit_and_skips_backend(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")
        job = make_cloud_job(dry_run=True)
        assert_dry_run_submit_contract(
            monkeypatch=monkeypatch,
            job=job,
        )

    def test_submit_stores_last_submitted_job(self, monkeypatch) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")
        job = make_cloud_job()
        submitted = SimpleNamespace(
            job_name="cloud-job-1",
            status=lambda: "Job status=Completed",
            logs=lambda **_: None,
        )

        monkeypatch.setattr(
            "job_submitter.jobs.base.CloudBackend.submit",
            lambda self, cfg: setattr(self, "last_submitted_job", submitted),
        )
        result = job.submit()

        assert result is job
        assert job._last_submitted_job is submitted

    def test_submit_raises_if_cloud_backend_returns_non_monitorable_handle(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")
        job = make_cloud_job()

        monkeypatch.setattr(
            "job_submitter.jobs.base.CloudBackend.submit",
            lambda self, cfg: setattr(self, "last_submitted_job", object()),
        )

        with pytest.raises(
            TypeError, match="Cloud backend did not return monitorable job handle"
        ):
            job.submit()

    def test_submit_raises_for_unknown_submission_type(self, monkeypatch) -> None:
        job = make_cloud_job()
        monkeypatch.setattr(
            job, "to_config", lambda: SimpleNamespace(submission_type="weird")
        )
        monkeypatch.setattr(job, "assert_job", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(job, "_before_submit", lambda: None)

        with pytest.raises(ValueError, match="Unknown submission_type=weird"):
            job.submit()

    def test_monitor_waits_and_captures_logs(self, monkeypatch) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")
        sleep_calls = []
        monkeypatch.setattr(
            "job_submitter.jobs.base.time.sleep",
            lambda seconds: sleep_calls.append(seconds),
        )

        class _SubmittedJob:
            job_name = "cloud-job-1"

            def __init__(self) -> None:
                self._statuses = [
                    "Job status=Pending",
                    "Job status=Running",
                    "Job status=Completed",
                ]
                self._idx = 0

            def status(self):
                if self._idx < len(self._statuses):
                    value = self._statuses[self._idx]
                    self._idx += 1
                    return value
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                assert tail == 0
                assert verbose is False
                print("test-log-line")
                return None

        job = make_cloud_job()
        submitted = _SubmittedJob()
        job._last_submitted_job = submitted
        monitor_result = job.monitor(
            initial_wait_seconds=1,
            poll_interval_seconds=2,
            logs_verbose=False,
            capture_logs=True,
            post_completion_wait_seconds=5,
        )

        assert sleep_calls == [1, 2, 2, 5, 2]
        assert monitor_result.final_status == "Job status=Completed"
        assert "test-log-line" in (monitor_result.logs or "")

    def test_monitor_retries_when_status_is_unavailable(self, monkeypatch) -> None:
        monkeypatch.setattr("job_submitter.jobs.base.time.sleep", lambda seconds: None)
        statuses = iter(
            [
                "Cant get status for job, check job_name",
                RuntimeError("HTTP 429 Rate limit exceeded"),
                "Job status=Pending",
                "Job status=Completed",
                RuntimeError("HTTP 429 Rate limit exceeded"),
            ]
        )

        class _SubmittedJob:
            job_name = "cloud-job-1"

            def status(self):
                value = next(statuses)
                if isinstance(value, Exception):
                    raise value
                return value

            def logs(self, tail=0, verbose=True):
                return ""

        job = make_cloud_job()
        job._last_submitted_job = _SubmittedJob()
        monitor_result = job.monitor(capture_logs=False)

        assert monitor_result.final_status == "Job status=Completed"

    def test_monitor_rejects_non_monitorable_job(self, monkeypatch) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")
        job = make_cloud_job()
        job._last_submitted_job = "submitted"

        with pytest.raises(TypeError):
            job.monitor()

    def test_extract_status_name_keeps_raw_status_without_prefix(self) -> None:
        assert DummyJob._extract_status_name("Completed") == "Completed"

    def test_capture_logs_returns_string_when_job_logs_returns_string(self) -> None:
        class _SubmittedJob:
            job_name = "cloud-job-1"

            def status(self):
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                assert tail == 7
                assert verbose is True
                return "captured-from-return"

        captured = DummyJob._capture_logs(
            job=_SubmittedJob(),
            logs_tail=7,
            logs_verbose=True,
        )
        assert captured == "captured-from-return"

    def test_capture_logs_merges_returned_and_printed_logs(self) -> None:
        class _SubmittedJob:
            def status(self):
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                assert tail == 3
                assert verbose is False
                print("printed-line")
                return "returned-line"

        captured = DummyJob._capture_logs(
            job=_SubmittedJob(),
            logs_tail=3,
            logs_verbose=False,
        )
        assert "printed-line" in captured
        assert "returned-line" in captured

    def test_capture_logs_prefers_captured_when_returned_is_subset(self) -> None:
        class _SubmittedJob:
            def status(self):
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                assert tail == 0
                assert verbose is True
                print("same-line")
                return "same-line"

        captured = DummyJob._capture_logs(
            job=_SubmittedJob(),
            logs_tail=0,
            logs_verbose=True,
        )
        assert captured == "same-line\n"

    def test_capture_logs_prefers_returned_when_contains_captured(self) -> None:
        class _SubmittedJob:
            def status(self):
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                assert tail == 0
                assert verbose is True
                print("same-line")
                return "same-line\nextended-tail"

        captured = DummyJob._capture_logs(
            job=_SubmittedJob(),
            logs_tail=0,
            logs_verbose=True,
        )
        assert captured == "same-line\nextended-tail"

    def test_monitor_capture_logs_empty_string(self, monkeypatch) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")

        class _SubmittedJob:
            job_name = "cloud-job-1"

            def status(self):
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                assert tail == 0
                assert verbose is False
                return ""

        job = make_cloud_job()
        job._last_submitted_job = _SubmittedJob()

        monitor_result = job.monitor(
            initial_wait_seconds=0,
            poll_interval_seconds=1,
            logs_verbose=False,
            capture_logs=True,
            logs_retry_attempts=1,
            post_completion_wait_seconds=0,
        )

        assert monitor_result.logs == ""
        assert monitor_result.final_status == "Job status=Completed"

    def test_monitor_raises_when_no_submitted_job_available(self, monkeypatch) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")
        job = make_cloud_job()
        job._last_submitted_job = None

        with pytest.raises(
            ValueError,
            match="No submitted cloud job to monitor. Call submit\\(\\) first",
        ):
            job.monitor()

    def test_monitor_raises_for_slurm_submission_type(self, monkeypatch) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "o1")
        job = make_slurm_job()

        with pytest.raises(
            AssertionError, match="Monitoring for slurm is not supported yet"
        ):
            job.monitor()

    def test_monitor_uses_last_submitted_job_without_wait_and_without_log_capture(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")
        sleep_calls = []
        monkeypatch.setattr(
            "job_submitter.jobs.base.time.sleep",
            lambda seconds: sleep_calls.append(seconds),
        )

        class _SubmittedJob:
            job_name = "cloud-job-1"

            def __init__(self) -> None:
                self.log_calls = 0
                self.status_calls = 0

            def status(self):
                self.status_calls += 1
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                self.log_calls += 1
                assert tail == 5
                assert verbose is True
                return None

        submitted = _SubmittedJob()
        job = make_cloud_job()
        job._last_submitted_job = submitted

        monitor_result = job.monitor(
            initial_wait_seconds=0,
            poll_interval_seconds=2,
            logs_tail=5,
            logs_verbose=True,
            capture_logs=False,
        )

        assert sleep_calls == []
        assert submitted.log_calls == 1
        assert submitted.status_calls >= 2
        assert monitor_result.logs is None
        assert monitor_result.final_status == "Job status=Completed"


@pytest.mark.unit
class TestBusinessScenariosEndToEnd:
    def test_end_to_end_cloud_submit_builds_business_config(self, monkeypatch) -> None:
        captured = {}
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")
        monkeypatch.setenv("PAM_PRIORITY_CLASS", "high")

        job = make_cloud_job(
            exp_name="online-rl-train",
            n_workers=2,
            n_gpus=4,
            extra_env={"FEATURE_FLAG": "1"},
        )

        def _fake_cloud_submit(self, cfg):
            captured["cfg"] = cfg
            self.last_submitted_job = SimpleNamespace(
                job_name="cloud-job-1",
                status=lambda: "Job status=Completed",
                logs=lambda **_: None,
            )

        monkeypatch.setattr(
            "job_submitter.jobs.base.CloudBackend.submit", _fake_cloud_submit
        )

        result = job.submit()
        cfg = captured["cfg"]
        assert result is job
        assert cfg.submission_type == "cloud"
        assert cfg.region == "SR008"
        assert cfg.queue_name == "shared"
        assert cfg.priority_class == "high"
        assert cfg.allocation_name == "alloc-sberdevices-giga-vision-h100-sr008"
        assert cfg.instance_type == "a100plus.4gpu.80vG.48C.728G"
        assert cfg.env_variables["FEATURE_FLAG"] == "1"
        assert cfg.job_desc == "u | online-rl-train"

    def test_end_to_end_slurm_submit_builds_business_config(self, monkeypatch) -> None:
        captured = {}
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "o1")
        monkeypatch.setenv("SLURM_JWT_TOKEN", "jwt")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820||http://b:6820")

        job = make_slurm_job(
            exp_name="slurm-train", queue_name="pretrain", n_workers=3, n_gpus=8
        )

        def _fake_submit(self, cfg):
            captured["cfg"] = cfg
            return None

        monkeypatch.setattr("job_submitter.jobs.base.SlurmBackend.submit", _fake_submit)

        result = job.submit()
        cfg = captured["cfg"]
        assert result is job
        assert cfg.submission_type == "slurm"
        assert cfg.region == "o1"
        assert cfg.partition == "pretrain"
        assert cfg.qos == "pretrain_low"
        assert cfg.account == "pretrain"
        assert cfg.instance_type is None
        assert cfg.slurm_jwt_token == "jwt"
        assert cfg.slurmrestd_urls == ["http://a:6820", "http://b:6820"]
        assert cfg.job_desc == "u | slurm-train"
        assert cfg.env_variables["NNODES"] == "3"
        assert cfg.env_variables["JS_GPUS_PER_NODE"] == "8"

    def test_end_to_end_dispatcher_submit_then_monitor(self, monkeypatch) -> None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")

        sleep_calls = []
        monkeypatch.setattr(
            "job_submitter.jobs.base.time.sleep",
            lambda seconds: sleep_calls.append(seconds),
        )

        status_job_names = []

        class FakeJob:
            def __init__(self, **kwargs):
                self.job_name = None

            def status(self):
                status_job_names.append(self.job_name)
                return "Job status=Completed"

            def logs(self, tail=0, verbose=True):
                return f"logs for {self.job_name}"

        monkeypatch.setattr(
            "job_submitter.core.backends.import_module",
            lambda _: SimpleNamespace(Job=FakeJob),
        )
        monkeypatch.setattr(
            "job_submitter.core.backends.requests.post",
            lambda *a, **kw: SimpleNamespace(
                status_code=200,
                text="ok",
                json=lambda: {"job_name": "lm-mpi-job-abc123"},
            ),
        )

        job = make_cloud_job(exp_name="e2e-dispatcher")
        job.submit()

        result = job.monitor(
            initial_wait_seconds=0,
            poll_interval_seconds=1,
            capture_logs=True,
            post_completion_wait_seconds=0,
            logs_retry_attempts=1,
        )

        assert status_job_names[0] == "lm-mpi-job-abc123"
        assert "logs for lm-mpi-job-abc123" in (result.logs or "")
        assert result.final_status == "Job status=Completed"

    def test_end_to_end_dispatcher_422_fallback_then_monitor(self, monkeypatch) -> None:
        """Regression: 7b2a781 overwrote job_name with submit() message on fallback."""
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")

        monkeypatch.setattr(
            "job_submitter.jobs.base.time.sleep",
            lambda _seconds: None,
        )

        status_job_names = []
        real_job_id = "lm-mpi-job-194b67e7-042a-4845-b4ca-3962008dcec0"

        class FakeJob:
            job_name = None

            def __init__(self, **_kwargs):
                return None

            def submit(self):
                self.job_name = real_job_id
                return f'Job "{self.job_name}" created.'

            def status(self):
                status_job_names.append(self.job_name)
                if self.job_name == real_job_id:
                    return "Job status=Completed"
                return "Cant get status for job, check job_name"

            def logs(self, tail=0, verbose=True):
                return f"logs for {self.job_name}"

        monkeypatch.setattr(
            "job_submitter.core.backends.import_module",
            lambda _: SimpleNamespace(Job=FakeJob),
        )
        monkeypatch.setattr(
            "job_submitter.core.backends.requests.post",
            lambda *a, **kw: SimpleNamespace(
                status_code=422,
                text='{"detail":"validation error"}',
                json=lambda: {"detail": "validation error"},
            ),
        )

        job = make_cloud_job(exp_name="e2e-dispatcher-422-fallback")
        job.submit()

        assert job._last_submitted_job is not None
        assert job._last_submitted_job.job_name == real_job_id
        assert "created." not in job._last_submitted_job.job_name

        result = job.monitor(
            initial_wait_seconds=0,
            poll_interval_seconds=1,
            capture_logs=False,
        )

        assert status_job_names[0] == real_job_id
        assert result.final_status == "Job status=Completed"


@pytest.mark.unit
class TestBusinessScenariosRegionalPolicy:
    @pytest.mark.parametrize(
        ("region", "expected_queue", "expected_allocation"),
        [
            ("SR004", None, None),
            ("SR006", None, None),
            ("SR008", "shared", "alloc-sberdevices-giga-vision-h100-sr008"),
        ],
    )
    def test_cloud_region_policy_matrix(
        self,
        monkeypatch,
        region: str,
        expected_queue: str | None,
        expected_allocation: str | None,
    ) -> None:
        monkeypatch.delenv("PAM_QUEUE_NAME", raising=False)
        monkeypatch.delenv("PAM_PRIORITY_CLASS", raising=False)
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", region)
        job = make_cloud_job(region=region)

        cfg = job.to_config()
        assert cfg.submission_type == "cloud"
        assert cfg.region == region
        assert cfg.queue_name == expected_queue
        assert cfg.allocation_name == expected_allocation
        if region == "SR008":
            assert cfg.priority_class is None
        else:
            assert cfg.priority_class == "medium"

    def test_tts_workspace_uses_tts_allocation_without_queue(self, monkeypatch) -> None:
        monkeypatch.delenv("PAM_PRIORITY_CLASS", raising=False)
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
        monkeypatch.setenv("CLUSTER_KEY", "SR008")
        monkeypatch.setenv("MLS_NTB_WORKSPACE_NAME", "wm-train")
        monkeypatch.setenv("PAM_QUEUE_NAME", "shared")
        job = make_cloud_job(
            region="SR008",
            queue_name="shared",
            workspace_name="tts",
        )

        cfg = job.to_config()

        assert cfg.queue_name is None
        assert cfg.priority_class is None
        assert cfg.allocation_name == "alloc-sberdevices-gigatts-h100-sr008"
        assert "PAM_QUEUE_NAME" not in cfg.env_variables
        assert cfg.env_variables["MLS_NTB_WORKSPACE_NAME"] == "tts"

    def test_slurm_priority_policy_controls_partition(self, monkeypatch) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "o1")
        monkeypatch.setenv("SLURM_JWT_TOKEN", "jwt")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820")
        job = make_slurm_job(queue_name="pretrain", priority_class="high")
        cfg = job.to_config()
        assert cfg.partition == "pretrain"

    def test_slurm_nice_is_reflected_in_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "o1")
        monkeypatch.setenv("SLURM_JWT_TOKEN", "jwt")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820")

        assert make_slurm_job(queue_name="pretrain").to_config().nice is None
        assert make_slurm_job(queue_name="pretrain", nice=20).to_config().nice == 20

    def test_slurm_excluded_nodes_are_reflected_in_config(self, monkeypatch) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "o1")
        monkeypatch.setenv("SLURM_JWT_TOKEN", "jwt")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820")
        job = make_slurm_job(queue_name="pretrain", exclude_nodes="node[001-002]")

        cfg = job.to_config()

        assert cfg.exclude_nodes == "node[001-002]"

    @pytest.mark.parametrize(
        ("region", "job_kwargs", "expected_gpu_type"),
        [
            ("o1", {"queue_name": "pretrain", "gpu_type": "a100"}, "a100"),
            ("o2", {"queue_name": "pretrain"}, "h100"),
        ],
    )
    def test_slurm_gpu_type_policy_is_reflected_in_config(
        self,
        monkeypatch,
        region: str,
        job_kwargs: dict[str, str],
        expected_gpu_type: str,
    ) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", region)
        monkeypatch.setenv("SLURM_JWT_TOKEN", "jwt")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820")
        job = make_slurm_job(region=region, **job_kwargs)

        cfg = job.to_config()

        assert cfg.submission_type == "slurm"
        assert cfg.region == region
        assert cfg.gpu_type == expected_gpu_type

    @pytest.mark.parametrize("region", ["o1", "o2"])
    def test_team_qos_config_carries_team_partition_qos_and_account(
        self, region, monkeypatch
    ) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", region)
        monkeypatch.setenv("SLURM_JWT_TOKEN", "jwt")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820")
        job = make_slurm_job(
            region=region, queue_name="alignment", priority_class="urgent"
        )

        cfg = job.to_config()

        assert cfg.queue_name == "alignment"
        assert cfg.partition == "alignment"
        assert cfg.qos == "alignment_high"
        assert cfg.account == "alignment"
        assert cfg.env_variables["PAM_QUEUE_NAME"] == "alignment"

    @pytest.mark.parametrize("region", ["o1", "o2"])
    def test_team_qos_config_without_queue_uses_slurm_default_account(
        self, region, monkeypatch
    ) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", region)
        monkeypatch.setenv("SLURM_JWT_TOKEN", "jwt")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820")
        monkeypatch.delenv("PAM_QUEUE_NAME", raising=False)
        monkeypatch.setattr(
            DummyJob, "_run_subprocess", staticmethod(lambda command: "data")
        )
        job = make_slurm_job(region=region)

        cfg = job.to_config()

        assert cfg.queue_name == "data"
        assert cfg.priority_class == "medium"
        assert cfg.partition == "data"
        assert cfg.qos == "data_low"
        assert cfg.account == "data"

    def test_legacy_slurm_config_has_no_qos_or_account_for_middle(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "o5")
        monkeypatch.setenv("SLURM_JWT_TOKEN", "jwt")
        monkeypatch.setenv("SLURMRESTD_URLS", "http://a:6820")
        job = make_slurm_job(region="o5")

        cfg = job.to_config()

        assert cfg.partition == "middle"
        assert cfg.qos is None
        assert cfg.account is None


@pytest.mark.unit
class TestBusinessScenariosAnonimizeLifecycle:
    def test_anonimize_config_lifecycle_transforms_config_in_place(
        self, tmp_path: Path
    ) -> None:
        slurm_job = make_slurm_job()
        chat_template = tmp_path / "chat.jinja"
        chat_template.write_text("plain text chat template", encoding="utf-8")
        cfg_path = tmp_path / "pam_lifecycle.yaml"
        OmegaConf.save(
            {
                "autometrics": {
                    "obm_proxy": {
                        "chat-template": str(chat_template),
                        "chat-template-kwargs": {"default_devsystem": "need-encode"},
                    },
                    "evaluators": {
                        "obm": {
                            "all": {"groups": ["sber-team"]},
                        }
                    },
                }
            },
            cfg_path,
        )

        original_cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        slurm_job._anonimize_config(cfg_path)

        updated_cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)

        # Config is edited in-place (no backup is created).
        assert original_cfg != updated_cfg
        assert not cfg_path.with_suffix(".backup.yaml").exists()

        # Updated config must contain transformed values.
        updated_chat = updated_cfg["autometrics"]["obm_proxy"]["chat-template"]
        assert updated_chat.endswith(".base64")
        assert Path(updated_chat).exists()
        assert updated_cfg["autometrics"]["evaluators"]["obm"]["all"]["groups"] == [
            "sber-team"
        ]
        assert (
            updated_cfg["autometrics"]["obm_proxy"]["chat-template-kwargs"][
                "default_devsystem"
            ]
            != "need-encode"
        )
