from pathlib import Path

import pytest
from job_submitter.asserts.autometrics import (
    AutometricsExternalAPIAssertor,
    AutometricsJobAssertor,
)
from job_submitter.core.job_config import JobTemplate
from omegaconf import OmegaConf


class _FakeAutoJob:
    def __init__(
        self,
        submission_type: str = "cloud",
        mode: str = "autometrics",
        region: str = "SR008",
    ):
        self.submission_type = submission_type
        self.mode = mode
        self.region = region
        self.user_login = "autotest"
        self.exp_name = "exp"
        self.n_workers = 1
        self.n_gpus = 8
        self.checkpoint_dir = "."
        self.path_to_config = None


def _external_api_config() -> dict:
    return {
        "run_name": "run",
        "exp_name": "exp",
        "user": "user",
        "autometrics": {
            "inference": {
                "engine_type": "external_api",
                "base_url": "https://example.com/v1",
                "model": "model",
                "api_key": "secret",
            },
            "evaluators": {"lm_eval": {}},
        },
    }


def _write_external_api_config(path: Path, config: dict) -> Path:
    OmegaConf.save(config, path)
    return path


@pytest.mark.unit
class TestAutometricsExternalAPIAssertor:
    def test_assert_job_accepts_valid_cloud_config_with_optional_inference_params(
        self, tmp_path: Path, cloud_template: JobTemplate
    ) -> None:
        config = _external_api_config()
        config["autometrics"]["inference"].update(
            {"formatting": "chat", "max_concurrency": 4}
        )
        config_path = _write_external_api_config(tmp_path / "config.yaml", config)
        job = _FakeAutoJob()
        job.path_to_config = str(config_path)

        AutometricsExternalAPIAssertor().assert_job(job, cloud_template)

    def test_assert_job_rejects_slurm_submission(
        self, tmp_path: Path, slurm_template: JobTemplate
    ) -> None:
        config_path = _write_external_api_config(
            tmp_path / "config.yaml", _external_api_config()
        )
        job = _FakeAutoJob(submission_type="slurm")
        job.path_to_config = str(config_path)

        with pytest.raises(AssertionError):
            AutometricsExternalAPIAssertor().assert_job(job, slurm_template)

    @pytest.mark.parametrize("value", [None, "", "   ", "...", 1])
    def test_non_empty_config_strings_reject_placeholders_and_non_strings(
        self, value: object
    ) -> None:
        with pytest.raises(AssertionError):
            AutometricsExternalAPIAssertor()._assert_non_empty_string(value, "key")

    def test_config_path_must_point_to_a_file(self, tmp_path: Path) -> None:
        assertor = AutometricsExternalAPIAssertor()

        with pytest.raises(AssertionError):
            assertor._assert_config_path(None)
        with pytest.raises(AssertionError):
            assertor._assert_config_path(str(tmp_path / "missing.yaml"))
        with pytest.raises(AssertionError):
            assertor._assert_config_path(str(tmp_path))

    @pytest.mark.parametrize("missing_key", ["engine_type", "base_url", "model"])
    def test_inference_requires_external_api_connection_fields(
        self, missing_key: str
    ) -> None:
        inference = _external_api_config()["autometrics"]["inference"]
        inference.pop(missing_key)

        with pytest.raises(AssertionError):
            AutometricsExternalAPIAssertor()._assert_inference(inference)

    def test_inference_accepts_user_pass_without_api_key_for_ift(self) -> None:
        inference = _external_api_config()["autometrics"]["inference"]
        inference.pop("api_key")
        inference["user_pass"] = "user:password"

        AutometricsExternalAPIAssertor()._assert_inference(inference)

    def test_inference_requires_api_key_for_openai_formatting(self) -> None:
        inference = _external_api_config()["autometrics"]["inference"]
        inference.pop("api_key")
        inference["user_pass"] = "user:password"
        inference["formatting"] = "openai"

        with pytest.raises(AssertionError):
            AutometricsExternalAPIAssertor()._assert_inference(inference)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("base_url", ""),
            ("model", "..."),
            ("api_key", None),
            ("formatting", "   "),
            ("max_concurrency", 0),
            ("max_concurrency", "4"),
        ],
    )
    def test_inference_rejects_invalid_field_values(
        self, key: str, value: object
    ) -> None:
        inference = _external_api_config()["autometrics"]["inference"]
        inference[key] = value

        with pytest.raises(AssertionError):
            AutometricsExternalAPIAssertor()._assert_inference(inference)

    @pytest.mark.parametrize(
        "inference",
        [
            None,
            {"engine_type": "sglang"},
            {
                "engine_type": "external_api",
                "base_url": "example.com/v1",
                "model": "model",
                "api_key": "secret",
            },
        ],
    )
    def test_inference_rejects_invalid_shape_engine_and_url(
        self, inference: object
    ) -> None:
        with pytest.raises(AssertionError):
            AutometricsExternalAPIAssertor()._assert_inference(inference)

    def test_inference_accepts_minimal_config(self) -> None:
        inference = _external_api_config()["autometrics"]["inference"]

        AutometricsExternalAPIAssertor()._assert_inference(inference)

    @pytest.mark.parametrize(
        "missing_key", ["run_name", "exp_name", "user", "autometrics"]
    )
    def test_external_api_config_requires_top_level_fields(
        self, tmp_path: Path, missing_key: str
    ) -> None:
        config = _external_api_config()
        config.pop(missing_key)
        config_path = _write_external_api_config(
            tmp_path / f"missing-{missing_key}.yaml", config
        )

        with pytest.raises(AssertionError):
            AutometricsExternalAPIAssertor()._assert_external_api_config(config_path)

    @pytest.mark.parametrize("autometrics", [{}, {"inference": None}])
    def test_external_api_config_requires_inference_and_evaluators(
        self, tmp_path: Path, autometrics: dict
    ) -> None:
        config = _external_api_config()
        config["autometrics"] = autometrics
        config_path = _write_external_api_config(tmp_path / "config.yaml", config)

        with pytest.raises(AssertionError):
            AutometricsExternalAPIAssertor()._assert_external_api_config(config_path)

    def test_external_api_config_requires_at_least_one_evaluator(
        self, tmp_path: Path
    ) -> None:
        config = _external_api_config()
        config["autometrics"]["evaluators"] = {}
        config_path = _write_external_api_config(tmp_path / "config.yaml", config)

        with pytest.raises(AssertionError):
            AutometricsExternalAPIAssertor()._assert_external_api_config(config_path)


@pytest.mark.unit
class TestAutometricsAssertorHelpers:
    def test_mode_validation(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_mode("autometrics")
        with pytest.raises(AssertionError):
            assertor._assert_mode("bad")

    def test_model_info_validation(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_model_info(
            {"model_class": "sft", "size": "pro", "username": "user"}
        )
        with pytest.raises(AssertionError):
            assertor._assert_model_info(
                {"model_class": "bad", "size": "pro", "username": "user"}
            )

    def test_inference_tp_forbidden_keys_validation(self) -> None:
        assertor = AutometricsJobAssertor()
        job = _FakeAutoJob()
        assertor._assert_inference_tp_and_forbidden_params({"tp": 8}, job)
        with pytest.raises(AssertionError):
            assertor._assert_inference_tp_and_forbidden_params(
                {"tp": 8, "ep-size": 1}, job
            )

    def test_is_obm_config_supports_string_and_dict(self) -> None:
        assertor = AutometricsJobAssertor()
        assert assertor._is_obm_config({"obm_config": "devices"}, "devices") is True
        assert assertor._is_obm_config({"obm_config": {"openai": {}}}, "openai") is True
        assert assertor._is_obm_config(None, "devices") is False

    def test_checkpoint_dir_validation(self, tmp_path: Path) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_checkpoint_dir(str(tmp_path))
        with pytest.raises(AssertionError):
            assertor._assert_checkpoint_dir("")

    def test_convertation_and_preparation_validations(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_convertation(
            {"mappers": ["map_gigafsdp_to_hf_safetensor"], "input_tp_size": 8}
        )
        assertor._assert_convertation(
            {
                "mappers": ["map_mixed_gigafsdp_to_mtp_safetensors"],
                "input_tp_size": 1,
            }
        )
        assertor._assert_convertation(
            {
                "mappers": ["map_mixed_gigafsdp_to_fsdp", "map_fsdp_to_hf"],
                "input_tp_size": 8,
            }
        )
        assertor._assert_convertation(
            {
                "mappers": ["map_safetensors_graft_mtp_safetensors"],
                "input_tp_size": 1,
                "mtp_donor_path": "/tmp/donor_hf",
            }
        )
        assertor._assert_preparation({"skip": True})
        with pytest.raises(AssertionError):
            assertor._assert_convertation({"mappers": ["bad"], "input_tp_size": 8})
        with pytest.raises(AssertionError):
            assertor._assert_convertation(
                {
                    "mappers": ["map_fsdp_to_hf", "map_mixed_gigafsdp_to_fsdp"],
                    "input_tp_size": 8,
                }
            )
        with pytest.raises(AssertionError):
            assertor._assert_convertation(
                {
                    "mappers": ["map_safetensors_graft_mtp_safetensors"],
                    "input_tp_size": 1,
                }
            )
        with pytest.raises(AssertionError):
            assertor._assert_preparation({"skip": False})

    def test_obm_evaluator_invalid_type_rejected(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_obm_evaluator({"obm_config": 1, "all": {}}, _FakeAutoJob())

    def test_obm_evaluator_with_mode_not_autometrics_early_return(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_obm_evaluator(
            {"obm_config": "devices", "all": {}}, _FakeAutoJob(mode="convertation")
        )
        assertor._assert_obm_evaluator(None, _FakeAutoJob())

    def test_obm_evaluator_dict_without_expected_keys_rejected(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_obm_evaluator(
                {"obm_config": {"other": {}}, "all": {}}, _FakeAutoJob()
            )

    def test_inference_workers_validation_errors(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_inference_workers(
                [1, 2],
                _FakeAutoJob(),
                {"callbacks": {"autometrics": {"nnodes": 3}}},
            )
        bad_job = _FakeAutoJob()
        bad_job.n_workers = 3
        with pytest.raises(AssertionError):
            assertor._assert_inference_workers(
                [1, 2], bad_job, {}, source_job="RayTrainJobVERL"
            )
        with pytest.raises(AssertionError):
            assertor._assert_inference_workers([1, 2], bad_job, {})

    def test_inference_tp_workers_and_ep_validation(self) -> None:
        assertor = AutometricsJobAssertor()
        job = _FakeAutoJob()
        job.n_workers = 3
        with pytest.raises(AssertionError):
            assertor._assert_inference_tp_workers(16, job, {})
        with pytest.raises(AssertionError):
            assertor._assert_inference_tp_and_forbidden_params(
                {"tp": 8, "ep": 4, "moe-a2a-backend": "deepep"}, _FakeAutoJob()
            )

    @pytest.mark.parametrize(
        "inference_params",
        [
            {"tp": 16, "ep": 1},
            {"tp": 16, "ep": 8},
        ],
    )
    def test_ep_may_differ_from_tp_without_deepep(self, inference_params: dict) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_inference_tp_and_forbidden_params(
            inference_params, _FakeAutoJob()
        )

    def test_inference_parallelism_dp_pp_and_tp_branches(self) -> None:
        assertor = AutometricsJobAssertor()
        job = _FakeAutoJob()
        job.n_workers = 2
        # dp branch
        assertor._assert_inference_parallelism({"tp": 4, "dp": 2}, job, {})
        # pp branch
        assertor._assert_inference_parallelism({"tp": 8, "pp": 2}, job, {})
        # tp > 8 branch
        assertor._assert_inference_parallelism({"tp": 16}, job, {})

    @pytest.mark.parametrize("dp", [1, 2, 4, 8])
    def test_dp_attention_accepts_every_dp_dividing_tp(self, dp: int) -> None:
        assertor = AutometricsJobAssertor()
        job = _FakeAutoJob()

        assertor._assert_inference_parallelism(
            {"tp": 8, "dp": dp, "enable-dp-attention": True}, job, {}
        )

    def test_dp_attention_rejects_dp_not_dividing_tp(self) -> None:
        assertor = AutometricsJobAssertor()
        job = _FakeAutoJob()

        with pytest.raises(AssertionError, match="must divide tp"):
            assertor._assert_inference_parallelism(
                {"tp": 8, "dp": 3, "enable-dp-attention": True}, job, {}
            )

    def test_instruct_requirements_reject_missing_strategy_for_obm(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_instruct_formatting_requirements(
                {"obm": {"obm_config": "devices", "all": {"groups": ["assistant"]}}},
                {"tp": 8},
                {},
            )

    def test_inference_convertation_mode_returns_early(self) -> None:
        assertor = AutometricsJobAssertor()
        job = _FakeAutoJob(mode="convertation")
        assertor._assert_inference({"tp": 8}, None, job, {})

    def test_inference_none_obm_and_openai_paths(self) -> None:
        assertor = AutometricsJobAssertor()
        job = _FakeAutoJob()
        assertor._assert_inference({"tp": 8}, None, job, {})
        assertor._assert_inference(
            {"tp": 8},
            {"obm_config": "openai", "all": {}},
            job,
            {},
        )

    def test_obm_proxy_slurm_no_kwargs_early_return(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_obm_proxy({}, _FakeAutoJob(submission_type="slurm"))

    def test_obm_proxy_slurm_empty_default_devsystem_skips_base64_check(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_obm_proxy(
            {"chat-template-kwargs": {"default_devsystem": ""}},
            _FakeAutoJob(submission_type="slurm"),
        )

    def test_obm_proxy_slurm_rejects_non_base64_chat_template(
        self, tmp_path: Path
    ) -> None:
        assertor = AutometricsJobAssertor()
        chat_template = tmp_path / "chat_template.jinja"
        chat_template.write_text("{{ user_message }}", encoding="utf-8")

        with pytest.raises(AssertionError):
            assertor._assert_obm_proxy(
                {"chat-template": str(chat_template)},
                _FakeAutoJob(submission_type="slurm"),
            )

    def test_obm_proxy_slurm_accepts_base64_chat_template(self, tmp_path: Path) -> None:
        assertor = AutometricsJobAssertor()
        chat_template = tmp_path / "chat_template.base64"
        chat_template.write_text("eysgdXNlcl9tZXNzYWdlIH19", encoding="utf-8")

        assertor._assert_obm_proxy(
            {"chat-template": str(chat_template)},
            _FakeAutoJob(submission_type="slurm"),
        )

    def test_instruct_requirements_no_instruct_metrics_early_return(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_instruct_formatting_requirements({}, {"tp": 8}, {})

    def test_instruct_requirements_returns_when_obm_absent(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_instruct_formatting_requirements(
            {"lm_eval": {"instruct": {"task": {}}}},
            {"tp": 8, "reasoning-parser": "r1"},
            {},
        )


@pytest.mark.unit
class TestAutometricsAssertorConfig:
    def test_pam_config_validation_for_cloud(self, tmp_path: Path) -> None:
        cfg = {
            "run_name": "run",
            "exp_name": "exp",
            "user": "u",
            "tensorboard_path": "/tmp/tb",
            "max_seq_len": 1024,
            "autometrics": {
                "globals": {"delete_converted_hf": True},
                "router": {"host": "localhost"},
                "convertation": {
                    "mappers": ["map_gigafsdp_to_hf"],
                    "input_tp_size": 8,
                },
                "preparation": {"skip": True},
                "model_info": {"model_class": "sft", "size": "pro", "username": "u"},
                "evaluators": {
                    "obm": {"obm_config": "openai", "all": {"groups": ["assistant"]}}
                },
                "inference": {
                    "tp": 8,
                    "reasoning-parser": "deepseek-r1",
                },
            },
        }

        assertor = AutometricsJobAssertor()
        assertor._assert_pam_config(cfg, _FakeAutoJob(submission_type="cloud"))

    def test_obm_proxy_requires_base64_default_devsystem_on_slurm(self) -> None:
        assertor = AutometricsJobAssertor()
        job = _FakeAutoJob(submission_type="slurm")
        with pytest.raises(AssertionError):
            assertor._assert_obm_proxy(
                {"chat-template-kwargs": {"default_devsystem": "not-base64"}},
                job,
            )

    def test_obm_evaluator_rejects_sber_groups_on_slurm(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_obm_evaluator(
                {"obm_config": "devices", "all": {"groups": ["sber"]}},
                _FakeAutoJob(submission_type="slurm"),
            )

    def test_obm_evaluator_allows_groups_with_sber_substring_on_slurm(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_obm_evaluator(
            {"obm_config": "devices", "all": {"groups": ["sber-team"]}},
            _FakeAutoJob(submission_type="slurm"),
        )

    def test_obm_evaluator_rejects_legacy_group_key_on_slurm(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_obm_evaluator(
                {"obm_config": "devices", "all": {"group": "SBER"}},
                _FakeAutoJob(submission_type="slurm"),
            )

    def test_obm_evaluator_slurm_allows_missing_group_and_default_devsystem(
        self,
    ) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_obm_evaluator(
            {"obm_config": "devices", "all": {"groups": []}},
            _FakeAutoJob(submission_type="slurm"),
        )

    def test_obm_evaluator_slurm_allows_new_groups_list(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_obm_evaluator(
            {"obm_config": "devices", "all": {"groups": ["assistant", "b2c"]}},
            _FakeAutoJob(submission_type="slurm"),
        )

    def test_obm_evaluator_slurm_rejects_non_dict_all(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_obm_evaluator(
                {"obm_config": "devices", "all": ["assistant"]},
                _FakeAutoJob(submission_type="slurm"),
            )

    def test_obm_evaluator_slurm_rejects_non_string_group_items(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_obm_evaluator(
                {"obm_config": "devices", "all": {"groups": ["assistant", 1]}},
                _FakeAutoJob(submission_type="slurm"),
            )

    def test_obm_evaluator_rejects_sber_when_groups_is_string_on_slurm(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_obm_evaluator(
                {"obm_config": "devices", "all": {"groups": "sber"}},
                _FakeAutoJob(submission_type="slurm"),
            )

    def test_obm_evaluator_rejects_uppercase_groups_on_slurm(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_obm_evaluator(
                {"obm_config": "devices", "all": {"groups": ["SBER"]}},
                _FakeAutoJob(submission_type="slurm"),
            )

    def test_instruct_requirements_allow_openai_with_obm_proxy_template(
        self, tmp_path: Path
    ) -> None:
        assertor = AutometricsJobAssertor()
        chat = tmp_path / "chat.jinja"
        chat.write_text("not-base64", encoding="utf-8")
        assertor._assert_instruct_formatting_requirements(
            {"obm": {"obm_config": "openai", "all": {"groups": ["assistant"]}}},
            {"tp": 8},
            {"chat-template": str(chat)},
        )

    def test_assert_autometrics_config_uses_default_path(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        assertor = AutometricsJobAssertor()
        ckpt = tmp_path / "ckpt"
        pam = ckpt / "artifacts" / "pam_config.yaml"
        pam.parent.mkdir(parents=True)
        pam.write_text("x: 1\n", encoding="utf-8")

        called = {}
        monkeypatch.setattr(
            assertor,
            "_load_config",
            lambda path: {
                "run_name": "r",
                "exp_name": "e",
                "user": "u",
                "tensorboard_path": "tb",
                "max_seq_len": 1,
                "autometrics": {
                    "globals": {"delete_converted_hf": True},
                    "router": {"host": "h"},
                    "convertation": {
                        "mappers": ["map_gigafsdp_to_hf"],
                        "input_tp_size": 1,
                    },
                    "preparation": {"skip": True},
                    "model_info": {
                        "model_class": "sft",
                        "size": "pro",
                        "username": "u",
                    },
                    "evaluators": {"obm": {"obm_config": "openai", "all": {}}},
                    "inference": {"tp": 8},
                },
            },
        )
        monkeypatch.setattr(
            assertor,
            "_assert_pam_config",
            lambda cfg, job, source_job="standalone": called.setdefault("ok", True),
        )

        job = _FakeAutoJob(submission_type="cloud")
        job.checkpoint_dir = str(ckpt)
        template = object()
        assertor._assert_autometrics_config(None, job, template)
        assert called["ok"] is True

    def test_assert_autometrics_config_accepts_explicit_path(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        assertor = AutometricsJobAssertor()
        pam = tmp_path / "pam_config.yaml"
        pam.write_text("x: 1\n", encoding="utf-8")
        called = {}

        monkeypatch.setattr(assertor, "_load_config", lambda path: {"autometrics": {}})
        monkeypatch.setattr(
            assertor,
            "_assert_pam_config",
            lambda cfg, job, source_job="standalone": called.setdefault("ok", True),
        )

        job = _FakeAutoJob(submission_type="cloud")
        template = object()
        assertor._assert_autometrics_config(str(pam), job, template)
        assert called["ok"] is True

    def test_assert_job_wrapper_calls_internal_checks(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        assertor = AutometricsJobAssertor()
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        job = _FakeAutoJob(submission_type="cloud")
        job.checkpoint_dir = str(ckpt)
        job.mode = "autometrics"
        job.user_login = "u"
        job.exp_name = "e"
        job.n_gpus = 8
        job.queue_name = "shared"

        template = type(
            "T",
            (),
            {
                "submission_type": "cloud",
                "user_login": "u",
                "exp_name": "e",
                "job_desc": "u | e",
                "base_image": "cr.ai.cloud.ru/a/b:c",
                "script": "echo 1",
                "n_workers": 1,
                "processes_per_worker": 1,
                "script_type": "binary",
                "region": "SR008",
                "partition": "middle",
                "queue_name": "shared",
                "cpu_only": False,
            },
        )()
        monkeypatch.setattr(
            assertor, "_assert_autometrics_config", lambda *args, **kwargs: None
        )
        assertor.assert_job(job, template)

    def test_pam_config_convertation_mode_skips_instruct_requirements(self) -> None:
        cfg = {
            "run_name": "run",
            "exp_name": "exp",
            "user": "u",
            "tensorboard_path": "/tmp/tb",
            "max_seq_len": 1024,
            "autometrics": {
                "globals": {"delete_converted_hf": True},
                "router": {"host": "localhost"},
                "convertation": {
                    "mappers": ["map_gigafsdp_to_hf"],
                    "input_tp_size": 8,
                },
                "preparation": {"skip": True},
                "model_info": {"model_class": "sft", "size": "pro", "username": "u"},
                "evaluators": {
                    "obm": {"obm_config": "devices", "all": {"groups": ["assistant"]}}
                },
                "inference": {"tp": 8},
            },
        }
        assertor = AutometricsJobAssertor()
        assertor._assert_pam_config(cfg, _FakeAutoJob(mode="convertation"))


@pytest.mark.unit
class TestAutometricsAssertorPdInference:
    def _pd_inference(self, **overrides) -> dict:
        params = {
            "shared_params": {"tp": 16, "ep": 16},
            "prefill": {"chunked-prefill-size": 32768},
            "decode": {"cuda-graph-max-bs": 64},
        }
        params.update(overrides)
        return params

    def _job(self, n_workers: int) -> _FakeAutoJob:
        job = _FakeAutoJob()
        job.n_workers = n_workers
        return job

    def test_pd_config_with_valid_split_passes(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_inference(
            self._pd_inference(), None, self._job(n_workers=4), {}
        )

    def test_pd_accepts_pipeline_parallelism_with_dp_attention(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_inference(
            self._pd_inference(
                shared_params={"enable-dp-attention": True},
                prefill={"tp": 2, "ep": 2, "dp": 2, "pp": 4},
                decode={"tp": 8, "ep": 8, "dp": 8},
            ),
            None,
            self._job(n_workers=2),
            {},
        )

    def test_pd_side_params_win_over_shared(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_inference(
            self._pd_inference(
                prefill={"tp": 8, "ep": 8},
                decode={"tp": 8, "ep": 8},
            ),
            None,
            self._job(n_workers=2),
            {},
        )

    def test_pd_rejects_workers_that_cannot_fit_both_sides(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_inference(
                self._pd_inference(), None, self._job(n_workers=2), {}
            )

    def test_pd_rejects_workers_without_valid_split(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_inference(
                self._pd_inference(), None, self._job(n_workers=5), {}
            )

    def test_pd_accepts_asymmetric_side_sizes(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_inference(
            self._pd_inference(prefill={"tp": 8, "ep": 8}),
            None,
            self._job(n_workers=3),
            {},
        )

    def test_pd_reuses_ep_equals_tp_rule_per_side(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError, match="ep should be equal to tp"):
            assertor._assert_inference(
                self._pd_inference(decode={"ep": 8, "moe-a2a-backend": "deepep"}),
                None,
                self._job(n_workers=4),
                {},
            )

    def test_pd_allows_ep_one_on_prefill_with_deepep_on_decode(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_inference(
            self._pd_inference(
                prefill={"ep": 1},
                decode={"moe-a2a-backend": "deepep"},
            ),
            None,
            self._job(n_workers=4),
            {},
        )

    def test_pd_reuses_forbidden_size_aliases_per_side(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_inference(
                self._pd_inference(prefill={"tp-size": 16}),
                None,
                self._job(n_workers=4),
                {},
            )

    def test_pd_requires_dict_sides(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_inference(
                self._pd_inference(prefill=None),
                None,
                self._job(n_workers=4),
                {},
            )

    @pytest.mark.parametrize("fraction", [0, 1, 1.5, "half"])
    def test_pd_rejects_invalid_prefill_fraction(self, fraction) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_inference(
                self._pd_inference(prefill_fraction=fraction),
                None,
                self._job(n_workers=4),
                {},
            )

    def test_pd_accepts_valid_prefill_fraction(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_inference(
            self._pd_inference(prefill_fraction=0.25),
            None,
            self._job(n_workers=4),
            {},
        )

    def test_pd_valid_workers_reach_callbacks_branch(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_inference(
            self._pd_inference(),
            None,
            self._job(n_workers=1),
            {"callbacks": {"autometrics": {"nnodes": 4}}},
        )
        with pytest.raises(AssertionError):
            assertor._assert_inference(
                self._pd_inference(),
                None,
                self._job(n_workers=1),
                {"callbacks": {"autometrics": {"nnodes": 3}}},
            )

    def test_pd_formatting_check_reads_decode_side_parsers(self) -> None:
        assertor = AutometricsJobAssertor()
        assertor._assert_instruct_formatting_requirements(
            {"lm_eval": {"instruct": {"task": {}}}},
            assertor._merged_pd_side_params(
                self._pd_inference(
                    shared_params={
                        "tp": 16,
                        "ep": 16,
                        "reasoning-parser": "deepseek-r1",
                    }
                ),
                "decode",
            ),
            {},
        )


@pytest.mark.unit
class TestPdWorkerValidation:
    def _job(self, n_workers: int) -> _FakeAutoJob:
        job = _FakeAutoJob()
        job.n_workers = n_workers
        return job

    def _pd(self, prefill: dict, decode: dict, **top) -> dict:
        return {"prefill": prefill, "decode": decode, **top}

    def _accepts(
        self, inference: dict, n_workers: int, config: dict | None = None
    ) -> bool:
        assertor = AutometricsJobAssertor()
        try:
            assertor._assert_inference(
                inference, None, self._job(n_workers), config or {}
            )
            return True
        except AssertionError:
            return False

    @pytest.mark.parametrize(
        ("prefill_side", "decode_side", "prefill_nnodes", "decode_nnodes"),
        [
            pytest.param(
                {"tp": 8, "ep": 8}, {"tp": 8, "ep": 8}, 1, 1, id="1+1_node_instances"
            ),
            pytest.param(
                {"tp": 16, "ep": 16},
                {"tp": 16, "ep": 16},
                2,
                2,
                id="2+2_node_instances",
            ),
            pytest.param(
                {"tp": 8, "ep": 8}, {"tp": 16, "ep": 16}, 1, 2, id="asymmetric_1+2"
            ),
            pytest.param(
                {"tp": 32, "ep": 32}, {"tp": 16, "ep": 16}, 4, 2, id="asymmetric_4+2"
            ),
            pytest.param(
                {"tp": 8, "ep": 8, "pp": 2},
                {"tp": 8, "ep": 8},
                2,
                1,
                id="pp_doubles_prefill_instance",
            ),
            pytest.param(
                {"tp": 4, "ep": 4, "dp": 2},
                {"tp": 8, "ep": 8},
                1,
                1,
                id="sub_node_tp_still_takes_whole_node",
            ),
        ],
    )
    def test_accepts_exactly_the_worker_counts_with_exact_partition(
        self,
        prefill_side: dict,
        decode_side: dict,
        prefill_nnodes: int,
        decode_nnodes: int,
    ) -> None:
        inference = self._pd(prefill_side, decode_side)
        for n_workers in range(1, 33):
            partition_exists = any(
                (n_workers - prefill_count * prefill_nnodes) > 0
                and (n_workers - prefill_count * prefill_nnodes) % decode_nnodes == 0
                for prefill_count in range(1, n_workers // prefill_nnodes + 1)
            )
            accepted = self._accepts(inference, n_workers)
            assert accepted == partition_exists, (
                f"n_workers={n_workers}: assertor={'accept' if accepted else 'reject'}, "
                f"но точное разбиение {prefill_nnodes}k+{decode_nnodes}m "
                f"{'существует' if partition_exists else 'не существует'}"
            )

    @pytest.mark.parametrize(
        ("n_workers", "accepted"),
        [
            pytest.param(2, False, id="two_nodes_cannot_fit_two_2node_sides"),
            pytest.param(3, False, id="three_nodes_odd_for_2node_instances"),
            pytest.param(4, True, id="minimal_valid_pd"),
            pytest.param(5, False, id="five_nodes_would_leave_afk_node"),
            pytest.param(6, True, id="one_prefill_two_decode_instances"),
        ],
    )
    def test_tp16_both_sides_corner_cases(self, n_workers: int, accepted: bool) -> None:
        inference = self._pd({"tp": 16, "ep": 16}, {"tp": 16, "ep": 16})
        assert self._accepts(inference, n_workers) is accepted

    def test_afk_node_is_rejected_not_silently_wasted(self) -> None:
        inference = self._pd({"tp": 32, "ep": 32}, {"tp": 16, "ep": 16})
        assert not self._accepts(inference, 7)
        assert self._accepts(inference, 6)
        assert self._accepts(inference, 8)

    def test_rejection_message_lists_valid_worker_counts(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError) as err:
            assertor._assert_inference(
                self._pd({"tp": 16, "ep": 16}, {"tp": 16, "ep": 16}),
                None,
                self._job(5),
                {},
            )
        message = str(err.value)
        assert "n_workers" in message
        assert "4" in message and "6" in message and "8" in message

    def _evaluators_config(self, evaluators: dict) -> dict:
        return {"autometrics": {"evaluators": evaluators}}

    @pytest.mark.parametrize(
        ("evaluators", "doubled"),
        [
            pytest.param(
                {"obm_judged": {"judges": {"pollux-judge-32b": ["task"]}}},
                True,
                id="obm_judged_local_judge",
            ),
            pytest.param(
                {"obm_judged": {"judges": {"openrouter:gpt-5": ["task"]}}},
                False,
                id="obm_judged_external_judge",
            ),
            pytest.param(
                {"new_obm_judged": {"judges": {"pollux-judge-32b": ["task"]}}},
                True,
                id="new_obm_judged_local_judge",
            ),
            pytest.param(
                {"gigametric_judged": {"judges": {"pollux-judge-32b": ["task"]}}},
                True,
                id="gigametric_judged_local_judge",
            ),
            pytest.param(
                {"tau2": {"user_model_name": "Qwen3-235B-A22B-Instruct-2507-FP8"}},
                True,
                id="tau2_local_user_model",
            ),
            pytest.param(
                {"tau2": {"user_model_name": "openai:gpt-5"}},
                False,
                id="tau2_external_user_model",
            ),
            pytest.param(
                {"tau3": {}},
                True,
                id="tau3_defaults_to_local_judge",
            ),
            pytest.param(
                {"tau3": {"user_model_name": "openrouter:claude"}},
                False,
                id="tau3_external_user_model",
            ),
            pytest.param(
                {"forge": {"judge": "pollux-judge-32b"}},
                True,
                id="forge_local_judge",
            ),
            pytest.param(
                {"forge": {"judge": "openrouter:gpt-5"}},
                False,
                id="forge_external_judge",
            ),
            pytest.param(
                {
                    "forge": {
                        "judge": "openrouter:gpt-5",
                        "groups": [{"judge": "pollux-judge-32b"}],
                    }
                },
                True,
                id="forge_local_group_judge",
            ),
            pytest.param(
                {"dredd": {"judge": {"model_name": "pollux-judge-32b"}}},
                False,
                id="dredd_is_scoring_phase_and_does_not_halve_nodes",
            ),
            pytest.param(
                {"lm_eval": {"instruct": {"task": {}}}}, False, id="no_judges"
            ),
        ],
    )
    def test_local_judges_double_the_worker_requirement(
        self, evaluators: dict, doubled: bool
    ) -> None:
        inference = self._pd({"tp": 16, "ep": 16}, {"tp": 16, "ep": 16})
        config = self._evaluators_config(evaluators)
        assert self._accepts(inference, 4, config) is not doubled
        assert self._accepts(inference, 8, config) is True

    def test_judged_pd_rejects_odd_workers(self) -> None:
        inference = self._pd({"tp": 8, "ep": 8}, {"tp": 8, "ep": 8})
        config = self._evaluators_config(
            {"obm_judged": {"judges": {"pollux-judge-32b": ["t"]}}}
        )
        assert not self._accepts(inference, 3, config)
        assert not self._accepts(inference, 5, config)
        assert self._accepts(inference, 4, config)
        assert self._accepts(inference, 6, config)
        assert not self._accepts(inference, 2, config)

    def test_judged_pd_validates_callbacks_nnodes_with_doubled_list(self) -> None:
        assertor = AutometricsJobAssertor()
        inference = self._pd({"tp": 16, "ep": 16}, {"tp": 16, "ep": 16})
        config = {
            **self._evaluators_config(
                {"obm_judged": {"judges": {"pollux-judge-32b": ["t"]}}}
            ),
            "callbacks": {"autometrics": {"nnodes": 8}},
        }
        assertor._assert_inference(inference, None, self._job(1), config)
        config["callbacks"]["autometrics"]["nnodes"] = 4
        with pytest.raises(AssertionError):
            assertor._assert_inference(inference, None, self._job(1), config)


@pytest.mark.unit
class TestFlatJudgedWorkerValidation:
    def _job(self, n_workers: int) -> _FakeAutoJob:
        job = _FakeAutoJob()
        job.n_workers = n_workers
        return job

    def _judged_config(self) -> dict:
        return {
            "autometrics": {
                "evaluators": {"obm_judged": {"judges": {"pollux-judge-32b": ["task"]}}}
            }
        }

    def _accepts(self, inference: dict, n_workers: int, config: dict) -> bool:
        assertor = AutometricsJobAssertor()
        try:
            assertor._assert_inference(inference, None, self._job(n_workers), config)
            return True
        except AssertionError:
            return False

    @pytest.mark.parametrize(
        ("inference", "n_workers", "accepted"),
        [
            pytest.param({"tp": 8, "ep": 8}, 1, False, id="tp8_single_node_rejected"),
            pytest.param({"tp": 8, "ep": 8}, 2, True, id="tp8_two_nodes_ok"),
            pytest.param({"tp": 8, "ep": 8}, 3, False, id="tp8_odd_rejected"),
            pytest.param({"tp": 8, "ep": 8}, 4, True, id="tp8_four_nodes_ok"),
            pytest.param(
                {"tp": 16, "ep": 16}, 2, False, id="tp16_half_smaller_than_instance"
            ),
            pytest.param(
                {"tp": 16, "ep": 16}, 4, True, id="tp16_half_fits_one_instance"
            ),
            pytest.param(
                {"tp": 16, "ep": 16}, 6, False, id="tp16_half_of_three_nodes_rejected"
            ),
            pytest.param(
                {"tp": 16, "ep": 16}, 8, True, id="tp16_half_fits_two_instances"
            ),
            pytest.param(
                {"tp": 4, "ep": 4, "dp": 2}, 1, False, id="tp4dp2_single_node_rejected"
            ),
            pytest.param(
                {"tp": 4, "ep": 4, "dp": 2}, 2, True, id="tp4dp2_two_nodes_ok"
            ),
        ],
    )
    def test_judged_flat_worker_counts(
        self, inference: dict, n_workers: int, accepted: bool
    ) -> None:
        assert self._accepts(inference, n_workers, self._judged_config()) is accepted

    def test_without_judges_flat_behavior_is_unchanged(self) -> None:
        assert self._accepts({"tp": 8, "ep": 8}, 1, {})
        assert self._accepts({"tp": 8, "ep": 8}, 3, {})

    def test_external_judge_does_not_halve_nodes(self) -> None:
        config = {
            "autometrics": {
                "evaluators": {"obm_judged": {"judges": {"openrouter:gpt-5": ["task"]}}}
            }
        }
        assert self._accepts({"tp": 8, "ep": 8}, 1, config)

    def test_judged_flat_validates_callbacks_nnodes(self) -> None:
        assertor = AutometricsJobAssertor()
        config = {**self._judged_config(), "callbacks": {"autometrics": {"nnodes": 4}}}
        assertor._assert_inference({"tp": 16, "ep": 16}, None, self._job(1), config)
        config["callbacks"]["autometrics"]["nnodes"] = 6
        with pytest.raises(AssertionError):
            assertor._assert_inference({"tp": 16, "ep": 16}, None, self._job(1), config)

    def test_rejection_message_lists_valid_worker_counts(self) -> None:
        assertor = AutometricsJobAssertor()
        with pytest.raises(AssertionError) as err:
            assertor._assert_inference(
                {"tp": 16, "ep": 16}, None, self._job(6), self._judged_config()
            )
        message = str(err.value)
        assert "n_workers" in message
        assert "4" in message and "8" in message and "12" in message


@pytest.mark.unit
class TestPdCoverageGaps:
    def test_unknown_evaluator_has_no_secondary_models(self) -> None:
        assertor = AutometricsJobAssertor()
        assert assertor._evaluator_secondary_model_names("lm_eval", {}) == []

    def test_forge_non_dict_group_entries_are_ignored(self) -> None:
        assertor = AutometricsJobAssertor()
        names = assertor._evaluator_secondary_model_names(
            "forge", {"judge": "openrouter:gpt-5", "groups": ["not-a-dict"]}
        )
        assert names == ["openrouter:gpt-5"]

    def test_pam_config_with_pd_inference_resolves_decode_parsers(self) -> None:
        cfg = {
            "run_name": "run",
            "exp_name": "exp",
            "user": "u",
            "tensorboard_path": "/tmp/tb",
            "max_seq_len": 1024,
            "autometrics": {
                "globals": {"delete_converted_hf": True},
                "router": {"host": "localhost"},
                "convertation": {
                    "mappers": ["map_gigafsdp_to_hf"],
                    "input_tp_size": 8,
                },
                "preparation": {"skip": True},
                "model_info": {"model_class": "sft", "size": "pro", "username": "u"},
                "evaluators": {"lm_eval": {"instruct": {"task": {}}}},
                "inference": {
                    "shared_params": {"tp": 16, "ep": 16},
                    "prefill": {},
                    "decode": {"reasoning-parser": "deepseek-r1"},
                },
            },
        }

        assertor = AutometricsJobAssertor()
        job = _FakeAutoJob(submission_type="cloud")
        job.n_workers = 4
        assertor._assert_pam_config(cfg, job)
